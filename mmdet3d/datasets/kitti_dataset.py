# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class KittiDataset(Det3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.数据集根目录的路径。
        ann_file (str): Path of annotation file.数据集根目录的路径。
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to []. 用于数据处理的流水线。默认为[]。
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).指定作为输入的传感器数据的模态。默认为 dict(use_lidar=True)。
        default_cam_key (str): The default camera name adopted.
            Defaults to 'CAM2'.采用的默认相机名称。默认为'CAM2'。
        load_type (str): Type of loading mode. Defaults to 'frame_based'.加载模式的类型。默认为'frame_based'。

            - 'frame_based': Load all of the instances in the frame.加载帧中的所有实例。
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.加载帧中的所有实例，并需要转换为FOV-based数据类型以支持基于图像的检测器。
            - 'fov_image_based': Only load the instances inside the default
              cam, and need to convert to the FOV-based data type to support
              image-based detector.仅加载默认相机内的实例，并需要转换为FOV-based数据类型以支持基于图像的检测器。
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:
            基于`box_type_3d`，数据集将将框封装为其原始格式，然后转换为`box_type_3d`。
            在该数据集中默认为'LiDAR'。可用选项包括：
            - 'LiDAR': Box in LiDAR coordinates. LiDAR坐标中的框
            - 'Depth': Box in depth coordinates, usually for indoor dataset.深度坐标中的框，通常用于室内数据集   
            - 'Camera': Box in camera coordinates.相机坐标中的框     
        filter_empty_gt (bool): Whether to filter the data with empty GT.是否过滤掉没有GT的数据
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True. 如果设置为True，在数据流水线处理后，空注释的示例将被删除，并且将在`__getitem__`中选择一个随机示例。
            默认为True。
        test_mode (bool): Whether the dataset is in test mode. 数据集是否处于测试模式。
            Defaults to False.默认为False。
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. 用于过滤无效预测框的点云范围。
            Defaults to [0, -40, -3, 70.4, 40, 0.0].
    """
    # TODO: use full classes of kitti
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), #调色板RGB
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255)]
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
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
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

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        ann_info = self._remove_dontcare(ann_info)
        # in kitti, lidar2cam = R0_rect @ Tr_velo_to_cam
        lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        gt_bboxes_3d = CameraInstance3DBoxes(
            ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                 np.linalg.inv(lidar2cam))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
