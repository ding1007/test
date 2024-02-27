custom_imports = dict(
    imports=[
        'mmdet3d.datasets.TJ4D_dataset','mmdet3d.evaluation.metrics.TJ4D_metric',
        'mmdet3d.models.voxel_encoders.pillar_encoder'], #只需要写python文件名
    allow_failed_imports=False)
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_TJ4D.py',
    '../_base_/datasets/TJ4D-3d-4class.py',
    '../_base_/schedules/cyclic-40e.py', 
    '../_base_/default_runtime.py'
]   

point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
backend_args = None
data_root = 'data/TJ4D/'
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'TJ4D_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(Car=5, Pedestrian=3, Cyclist=4,Truck= 6)),
    classes = ['Car','Pedestrian', 'Cyclist', 'Truck'],
    sample_groups=dict(Car = 10, Pedestrian= 5, Cyclist=10,Truck= 5),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=[0,1,2,3,5],
        backend_args=backend_args),
    backend_args=backend_args)


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=[0,1,2,3,5],
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,  # 替换成您的点云数据维度
        use_dim=[0,1,2,3,5],
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# 为可视化阶段的数据和 GT 加载构造流水线
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=8,
        type='LoadPointsFromFile',
        use_dim=[0,1,2,3,5]),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]

# train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
# val_dataloader =  dict(dataset=dict(pipeline=eval_pipeline))
# test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
info_prefix ='TJ4Dv2'
train_dataloader = dict(dataset=dict(dataset=
                            dict(pipeline=train_pipeline,ann_file=info_prefix +'_infos_train.pkl')
                                     ))
val_dataloader =  dict(dataset=dict(pipeline=eval_pipeline,
                                ann_file=info_prefix +'_infos_val.pkl')
                                )
test_dataloader = dict(dataset=dict(pipeline=test_pipeline,
                                    ann_file=info_prefix +'_infos_test.pkl')
                                )
val_evaluator = dict(ann_file=data_root + info_prefix +'_infos_val.pkl')

voxel_encoder = dict(
    type='PillarFeatureNet',
    in_channels=5,
    feat_channels=[64],
    with_distance=False,
    voxel_size=[0.16, 0.16, 6],
    point_cloud_range=point_cloud_range)

randomness = dict(
    # seed= 104156657
    seed= 3047
)
visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')],name='tj4dv2')
model = dict(voxel_encoder=voxel_encoder)
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
epoch_num = 80
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs. 
# 重复因子是指在每个周期内重复使用相同的数据两次。
#TODO
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()