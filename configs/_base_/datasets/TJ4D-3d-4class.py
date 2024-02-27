# 数据集设置
dataset_type = 'TJ4DDataset'
data_root = 'data/TJ4D/'
class_names = ['Car','Pedestrian', 'Cyclist', 'Truck']  # 替换成您的数据集类别
point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]  # 根据您的数据集进行调整
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None

#数据增强pipiline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,  # 替换成您的点云数据维度
        use_dim=8,
        backend_args=backend_args),  # 替换成在训练和推理时实际使用的维度
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict( #打包放入字典
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,  # 替换成您的点云数据维度
        use_dim=8,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# 为可视化阶段的数据和 GT 加载构造流水线
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=8,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8, # 8个工作进程进行数据加载和预处理。
    persistent_workers=True,# 持久化工作线程，用于加速数据加载
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='TJ4D_infos_train.pkl',  # 指定您的训练pkl信息
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='TJ4D_infos_val.pkl',  # 指定您的验证 pkl 信息
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='TJ4D_infos_val.pkl',  # 指定您的验证 pkl 信息
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='TJ4DMetric',
    ann_file=data_root + 'TJ4D_infos_val.pkl',  # 指定您的验证 pkl 信息
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')