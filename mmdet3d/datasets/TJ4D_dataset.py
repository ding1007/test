from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset

@DATASETS.register_module()
class TJ4DDataset(Det3DDataset):
    # 替换成自定义 pkl 信息文件里的所有类别
    METAINFO = {
        'classes': ('Car','Pedestrian', 'Cyclist', 'Truck'),
        'palette': [(106, 0, 228), (119, 11, 32), 
                    (165, 42, 42), (0, 0, 192)] #调色板RGB

    }
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -4, 70.4, 40, 2],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

    def parse_data_info(self, info: dict) -> dict:
            """Process the raw data info.

            The only difference with it in `Det3DDataset`
            is the specific process for `plane`.

            Args:
                info (dict): Raw info dict.

            Returns:
                dict: Has `ann_info` in training stage. And
                all path has been converted to absolute path.
            """
            if self.modality['use_lidar']:
                if 'plane' in info:
                    # convert ground plane to velodyne coordinates
                    plane = np.array(info['plane'])
                    lidar2cam = np.array(
                        info['images']['CAM2']['lidar2cam'], dtype=np.float32)
                    reverse = np.linalg.inv(lidar2cam)

                    (plane_norm_cam, plane_off_cam) = (plane[:3],
                                                    -plane[:3] * plane[3])
                    plane_norm_lidar = \
                        (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                    plane_off_lidar = (
                        reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                        reverse[:3, 3])
                    plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
                    plane_lidar[:3] = plane_norm_lidar
                    plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
                else:
                    plane_lidar = None

                info['plane'] = plane_lidar
            #这段代码是在特定加载类型为'fov_image_based'
                #且允许加载评估注释 (self.load_eval_anns为真) 的情况下执行的。
            if self.load_type == 'fov_image_based' and self.load_eval_anns:
                info['instances'] = info['cam_instances'][self.default_cam_key]

            info = super().parse_data_info(info)

            return info


    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # 空实例
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # 过滤掉没有在训练中使用的类别
        ann_info = self._remove_dontcare(ann_info)
        # in kitti, lidar2cam = R0_rect @ Tr_velo_to_cam
        lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        gt_bboxes_3d = CameraInstance3DBoxes(
            ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                 np.linalg.inv(lidar2cam))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info