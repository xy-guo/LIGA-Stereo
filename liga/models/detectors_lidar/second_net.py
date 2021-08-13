# Modified from OpenPCDet. https://github.com/open-mmlab/OpenPCDet


from .lidar_detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.model_cfg.get('RETURN_BATCH_DICT', False):
            keys_to_remove = ['encoded_spconv_tensor',
                              'encoded_spconv_tensor_stride',
                              'multi_scale_3d_features',
                              'reg_features',
                              'box_cls_labels',
                              'box_reg_targets',
                              'reg_weights',
                              'anchors',
                              'cls_preds_normalized']
            # 'voxels', 'voxel_coords', 'voxel_num_points', 'voxel_features']
            for k in keys_to_remove:
                batch_dict.pop(k, None)
            keys_to_keep = ['spatial_features_stride',
                            'spatial_features',
                            'spatial_features_2d',
                            'volume_features',
                            'spatial_features_2d_prehg',
                            'batch_cls_preds',
                            'batch_box_preds']
            batch_dict['lidar_outputs'] = {}
            for k in keys_to_keep:
                if k in batch_dict:
                    batch_dict['lidar_outputs'][k] = batch_dict.pop(k)

            return batch_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
