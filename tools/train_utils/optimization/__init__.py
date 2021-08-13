from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


class FusedOptimizer(optim.Optimizer):
    def __init__(self, all_params, lr=None, weight_decay=None, momentum=None):
        self.optimizers = []
        ops = set([x['optimizer'] for x in all_params])
        for op in ops:
            params = [x for x in all_params if x['optimizer'] == op]
            if op == 'adam':
                optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            elif op == 'adamw':
                optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            elif op == 'sgd':
                optimizer = optim.SGD(
                    params, lr=lr, weight_decay=weight_decay,
                    momentum=momentum
                )
            else:
                raise ValueError('wrong ops type')
            self.optimizers.append(optimizer)

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(FusedOptimizer, self).__init__(all_params, defaults)

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def build_optimizer(model, optim_cfg):
    if getattr(optim_cfg, 'PER_PARAMETER_CFG', None) is None:
        params = [x for x in model.parameters() if x.requires_grad]
    else:
        all_parameters = dict(model.named_parameters())
        all_parameters = {k: v for k, v in all_parameters.items() if v.requires_grad}
        params = []

        for cur_cfg in optim_cfg.PER_PARAMETER_CFG:
            cur_params = []
            for k in list(all_parameters.keys()):
                if cur_cfg.START_WITH == 'others':
                    check_ok = True
                elif isinstance(cur_cfg.START_WITH, str):
                    check_ok = k.startswith(cur_cfg.START_WITH)
                elif isinstance(cur_cfg.START_WITH, list):
                    check_ok = any([k.startswith(start_str) for start_str in cur_cfg.START_WITH])
                else:
                    raise ValueError('wrong start_with config')
                if check_ok:
                    cur_params.append(all_parameters[k])
                    all_parameters.pop(k)
            assert len(cur_params) > 0, 'cannot find any parameter starting with {}'.format(cur_cfg.START_WITH)
            print(f"find {len(cur_params)} parameters starting with {cur_cfg.START_WITH}")
            params.append({
                "params": cur_params,
                "lr": optim_cfg.LR * cur_cfg.MUL_LR,
            })
            if 'optimizer' in cur_cfg:
                params[-1]['optimizer'] = cur_cfg.optimizer
        if len(all_parameters) > 0:
            print(f"find {len(all_parameters)} parameters left")
            print(list(all_parameters.keys()))

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'fused':
        optimizer = FusedOptimizer(params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY, momentum=optim_cfg.MOMENTUM)
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        assert getattr(optim_cfg, 'PER_CHILD_CFG', None) is None

        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        def flatten_model(m):
            return sum(map(flatten_model, m.children()), []) if num_children(m) else [m]

        def get_layer_groups(m):
            return [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * total_iters_each_epoch,
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
