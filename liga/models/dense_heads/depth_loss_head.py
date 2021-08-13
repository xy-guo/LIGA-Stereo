# Depth Loss Head for stereo matching supervision.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DepthLossHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.depth_loss_type = model_cfg.LOSS_TYPE
        self.loss_weights = model_cfg.WEIGHTS
        self.point_cloud_range = point_cloud_range
        self.min_depth = point_cloud_range[0]
        self.max_depth = point_cloud_range[3]
        self.forward_ret_dict = {}

    def get_loss(self, batch_dict, tb_dict=None):
        if tb_dict is None:
            tb_dict = {}

        depth_preds = batch_dict['depth_preds']
        depth_volumes = batch_dict['depth_volumes']
        depth_sample = batch_dict['depth_samples']
        # depth_fgmask_imgs = batch_dict['depth_fgmask_img'].squeeze(1)
        # gt_boxes_2d = batch_dict['gt_boxes_2d']
        gt = batch_dict['depth_gt_img'].squeeze(1)

        depth_loss = 0.
        assert len(depth_preds) == len(depth_volumes)
        assert len(depth_preds) == len(self.loss_weights)
        mask = (gt > self.min_depth) & (gt < self.max_depth)
        gt = gt[mask]
        depth_interval = depth_sample[1] - depth_sample[0]

        assert len(depth_preds) == len(depth_volumes)
        assert len(depth_volumes) == len(self.loss_weights)

        for i, (depth_pred, depth_cost, pred_weight) in enumerate(zip(depth_preds, depth_volumes, self.loss_weights)):
            depth_pred = depth_pred[mask]
            depth_cost = depth_cost.permute(0, 2, 3, 1)[mask]

            for loss_type, loss_type_weight in self.depth_loss_type.items():
                if depth_pred.shape[0] == 0:
                    print('no gt warning')
                    loss = depth_preds[i].mean() * 0.0
                else:
                    if loss_type == "l1":
                        loss = F.smooth_l1_loss(depth_pred, gt, reduction='none')
                        loss = loss.mean()
                    elif loss_type == "purel1":
                        loss = F.l1_loss(depth_pred, gt, reduction='none')
                        loss = loss.mean()
                    elif loss_type == "ce":
                        depth_log_prob = F.log_softmax(depth_cost, dim=1)
                        distance = torch.abs(
                            depth_sample.cuda() - gt.unsqueeze(-1)) / depth_interval
                        probability = 1 - distance.clamp(max=1.0)
                        loss = -(probability * depth_log_prob).sum(-1)
                        loss = loss.mean()
                    elif loss_type.startswith("gaussian"):
                        depth_log_prob = F.log_softmax(depth_cost, dim=1)
                        distance = torch.abs(
                            depth_sample.cuda() - gt.unsqueeze(-1))
                        sigma = float(loss_type.split("_")[1])
                        if dist.get_rank() == 0:
                            print("depth loss using gaussian normalized", sigma)
                        probability = torch.exp(-0.5 * (distance ** 2) / (sigma ** 2))
                        probability /= torch.clamp(probability.sum(1, keepdim=True), min=1.0)
                        loss = -(probability * depth_log_prob).sum(-1)
                        loss = loss.mean()
                    elif loss_type.startswith("laplacian"):
                        depth_log_prob = F.log_softmax(depth_cost, dim=1)
                        distance = torch.abs(
                            depth_sample.cuda() - gt.unsqueeze(-1))
                        sigma = float(loss_type.split("_")[1])
                        if dist.get_rank() == 0:
                            print("depth loss using laplacian normalized", sigma)
                        probability = torch.exp(-distance / sigma)
                        probability /= torch.clamp(probability.sum(1, keepdim=True), min=1.0)
                        loss = -(probability * depth_log_prob).sum(-1)
                        loss = loss.mean()
                    elif loss_type == "hard_ce":
                        depth_log_prob = F.log_softmax(depth_cost, dim=1)
                        distance = torch.abs(
                            depth_sample.cuda() - gt.unsqueeze(-1)) / depth_interval
                        probability = 1 - distance.clamp(max=1.0)
                        probability[probability >= 0.5] = 1.0
                        probability[probability < 0.5] = .0

                        loss = -(probability * depth_log_prob).sum(-1)

                        loss = loss.mean()
                    else:
                        raise NotImplementedError

                tb_dict['loss_depth_{}_{}'.format(i, loss_type)] = loss.item()
                depth_loss += pred_weight * loss_type_weight * loss

        return depth_loss, tb_dict

    def forward(self, batch_dict):
        if batch_dict['depth_preds'][-1].shape[0] != 1:
            raise NotImplementedError

        if not self.training:
            # depth_pred = batch_dict['depth_preds'][-1]
            depth_pred_local = batch_dict['depth_preds_local'][-1]
            # depth_cost = batch_dict['depth_volumes'][0].permute(0, 2, 3, 1)
            # depth_sample = batch_dict['depth_samples']
            gt = batch_dict['depth_gt_img'].squeeze(1)

            mask = (gt > self.min_depth) & (gt < self.max_depth)
            # depth_interval = depth_sample[1] - depth_sample[0]
            assert mask.sum() > 0

            # abs error
            error_map = torch.abs(depth_pred_local - gt) * mask.float()
            batch_dict['depth_error_map'] = error_map

            # mean_error = error_map[mask].mean()
            median_error = error_map[mask].median()

            # batch_dict['depth_error_local_mean'] = mean_error
            batch_dict['depth_error_all_local_median'] = median_error
            for thresh in [0.2, 0.4, 0.8, 1.6]:
                batch_dict[f"depth_error_all_local_{thresh:.1f}m"] = (error_map[mask] > thresh).float().mean()

            if 'depth_fgmask_img' in batch_dict:
                fg_mask = (gt > self.min_depth) & (gt < self.max_depth) & (batch_dict['depth_fgmask_img'].squeeze(1) > 0)
                local_errs = torch.abs(depth_pred_local - gt)
                fg_local_errs = local_errs[fg_mask]

                # fg local depth errors per instance
                fg_gts = gt[fg_mask]
                batch_dict['depth_error_fg_local_statistics_perbox'] = []
                fg_ids = batch_dict['depth_fgmask_img'].squeeze(1)[fg_mask].int() - 1
                if len(fg_ids) > 0:
                    for idx in range(fg_ids.min().item(), fg_ids.max().item() + 1):
                        if batch_dict['gt_index'][0][idx] < 0:
                            continue
                        if torch.sum(fg_ids == idx) <= 5:
                            continue
                        errs_i = fg_local_errs[fg_ids == idx]
                        fg_gt_i_median = fg_gts[fg_ids == idx].median().item()
                        num_points_i = (fg_ids == idx).sum().item()
                        batch_dict['depth_error_fg_local_statistics_perbox'].append(dict(
                            distance=fg_gt_i_median,
                            err_median=errs_i.median().item(),
                            num_points=num_points_i,
                            name=batch_dict['gt_names'][0][idx],
                            truncated=batch_dict['gt_truncated'][0][idx],
                            occluded=batch_dict['gt_occluded'][0][idx],
                            difficulty=batch_dict['gt_difficulty'][0][idx],
                            index=batch_dict['gt_index'][0][idx],
                            idx=idx,
                            image_idx=batch_dict['image_idx'][0]
                        ))

                        for thresh in [0.2, 0.4, 0.8, 1.6]:
                            batch_dict['depth_error_fg_local_statistics_perbox'][-1][f"err_{thresh:.1f}m"] = (errs_i > thresh).float().mean().item()

        return batch_dict
