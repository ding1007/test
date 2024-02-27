# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple
import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch import Tensor, nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from .utils import PFNLayer, AttenPFNLayer1, AttenPFNLayer2, get_paddings_indicator


@MODELS.register_module()
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels: Optional[int] = 5,
                 feat_channels: Optional[tuple] = (64, ), # ()元组形式，单个要加，
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 with_slide_window: Optional[bool] =False,

                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        #TODO: 修改输入参数
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        if with_slide_window:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self._with_slide_window = with_slide_window
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.x_l = (point_cloud_range[3] - point_cloud_range[0] )/ voxel_size[0]
        self.y_l = (point_cloud_range[4] - point_cloud_range[1] )/ voxel_size[1]
        self.z_l =( point_cloud_range[5] - point_cloud_range[2] )/ voxel_size[2]
    def sliding_window_count_torch(self,tensor, window_size):
        # 将窗口大小转换为kernel大小
        tensor = tensor.to(torch.float32)
        kernel_size = (window_size, window_size)
        # 构建一个全为1的kernel
        kernel = torch.ones(1, 1, *kernel_size, dtype=tensor.dtype, device=tensor.device)

        # 使用conv2d进行滑窗统计
        #TODO:
        counts = F.conv2d(tensor.unsqueeze(0), kernel, stride=1, padding=2).squeeze(0).squeeze(0)

        return counts
    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """

        features_ls = [features]
        # Find distance of x, y, and z from cluster center TODO:修改平均数
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        # 统计附近点的个数
        if self._with_slide_window:
            bs = coors[-1, 0] + 1
            np_features = torch.zeros(num_points.shape).to(features.device)
            for i in range(bs):
                cur_coors_idx = coors[:, 0] == i  #建一个布尔索引 cur_coors_idx，用于筛选批次索引等于当前循环迭代的 i 的样本。
                cur_coors = coors[cur_coors_idx, :]
                np_nums = num_points.view(-1, 1)[cur_coors_idx]
                np_tmp = torch.zeros((int(self.x_l), int(self.y_l),int(self.z_l)), dtype=torch.int,device=features.device)
                np_tmp[cur_coors[:, 1], cur_coors[:, 2]] = np_nums
                #TODO:
                np_tmp = self.sliding_window_count_torch(np_tmp.permute(2, 0, 1), 4) / 16
                np_nums_sum = np_tmp[cur_coors[:, 1], cur_coors[:, 2]]
                np_features[cur_coors_idx] = np_nums_sum
            np_features = np_features.unsqueeze(1).expand(-1, 32).unsqueeze(-1)
            features_ls.append(np_features)
        #Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that （pillars,32,features）32的位置，根据空的柱状体的点数位置将相应的特征置零
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask #特征置0

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)


@MODELS.register_module()
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.

    The network prepares the pillar features and performs forward pass
    through PFNLayers. The main difference is that it is used for
    dynamic voxels, which contains different number of points inside a voxel
    without limits.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels: Optional[int] = 4,
                 feat_channels: Optional[tuple] = (64, ),
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(DynamicPillarFeatureNet, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy)
        feat_channels = [self.in_channels] + list(feat_channels)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """Map the centers of voxels to its corresponding points.

        Args:
            pts_coors (torch.Tensor): The coordinates of each points, shape
                (M, 3), where M is the number of points.
            voxel_mean (torch.Tensor): The mean or aggregated features of a
                voxel, shape (N, C), where N is the number of voxels.
            voxel_coors (torch.Tensor): The coordinates of each voxel.

        Returns:
            torch.Tensor: Corresponding voxel centers of each points, shape
                (M, C), where M is the number of points.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def forward(self, features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors
@MODELS.register_module()
class AttentionPillarFeatureNet(nn.Module):
    def __init__(self,
                 in_channels: Optional[int] = 5,
                 feat_channels: Optional[tuple] = (64, ), # ()元组形式，单个要加，
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 with_slide_window: Optional[bool] =False,

                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(AttentionPillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        #TODO: 修改输入参数
        if with_cluster_center:
            in_channels += 4
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        if with_slide_window:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self._with_slide_window = with_slide_window
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers1 = []
        pfn_layers2 = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers1.append(
                AttenPFNLayer1(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
            pfn_layers2.append(
                AttenPFNLayer2(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode='avg'))
        self.pfn_layers1 = nn.ModuleList(pfn_layers1)
        self.pfn_layers2 = nn.ModuleList(pfn_layers2)
        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.x_l = (point_cloud_range[3] - point_cloud_range[0] )/ voxel_size[0]
        self.y_l = (point_cloud_range[4] - point_cloud_range[1] )/ voxel_size[1]
        self.z_l = (point_cloud_range[5] - point_cloud_range[2] )/ voxel_size[2]
    def sliding_window_count_torch(self,tensor, window_size):
        # 将窗口大小转换为kernel大小
        tensor = tensor.to(torch.float32)
        kernel_size = (window_size, window_size)
        # 构建一个全为1的kernel
        kernel = torch.ones(1, 1, *kernel_size, dtype=tensor.dtype, device=tensor.device)

        # 使用conv2d进行滑窗统计
        #TODO:
        counts = F.conv2d(tensor.unsqueeze(0), kernel, stride=1, padding=2).squeeze(0).squeeze(0)

        return counts
    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """

        features_ls = [features]
        # Find distance of x, y, and z from cluster center TODO:修改平均数
        if self._with_cluster_center:
            points_mean = features[:, :, :4].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :4] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        # 统计附近点的个数
        if self._with_slide_window:
            bs = coors[-1, 0] + 1
            np_features = torch.zeros(num_points.shape).to(features.device)
            for i in range(bs):
                cur_coors_idx = coors[:, 0] == i  #建一个布尔索引 cur_coors_idx，用于筛选批次索引等于当前循环迭代的 i 的样本。
                cur_coors = coors[cur_coors_idx, :]
                np_nums = num_points.view(-1, 1)[cur_coors_idx]
                np_tmp = torch.zeros((int(self.x_l), int(self.y_l),int(self.z_l)), dtype=torch.int,device=features.device)
                np_tmp[cur_coors[:, 1], cur_coors[:, 2]] = np_nums
                #TODO:
                np_tmp = self.sliding_window_count_torch(np_tmp.permute(2, 0, 1), 4) / 16
                np_nums_sum = np_tmp[cur_coors[:, 1], cur_coors[:, 2]]
                np_features[cur_coors_idx] = np_nums_sum
            np_features = np_features.unsqueeze(1).expand(-1, 32).unsqueeze(-1)
            features_ls.append(np_features)
        #Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that （pillars,32,features）32的位置，根据空的柱状体的点数位置将相应的特征置零
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask #特征置0

        for pfn in self.pfn_layers1:
            x,x_max1 = pfn(features, num_points)
        for pfn in self.pfn_layers2:
            x_max2 = pfn(features, num_points,x)
        features = (x_max1+x_max2)/2.0
        return features.squeeze(1)