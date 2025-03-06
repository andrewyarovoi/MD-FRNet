# For TARGET, we reduced the classes to just 8 to simplify the problem and improve performance
dataset_type = 'MixedDataset'
class_names = [
    'ground', 'road', 'vegetation', 'structure', 'vehicle', 'humans', 'object', 'outliers']
input_modality = dict(use_lidar=True, use_camera=False)

# define waymo config
waymo_data_root = '/mnt/d/waymo/kitti_format/'
waymo_labels_map = {
    0: 8,   # "undefined" mapped to "unlabled"
    1: 4,   # "car" mapped to "vehicle"
    2: 4,   # "truck" mapped to "vehicle"
    3: 4,   # "bus" mapped to "vehicle"
    4: 4,   # "other_vehicle" mapped to "vehicle"
    5: 4,   # "other_vehicle" mapped to "vehicle"
    6: 5,   # "bicyclist" mapped to "humans"
    7: 5,   # "pedestrian" mapped to "humans"
    8: 6,   # "sign" mapped to "object"
    9: 6,   # "trafic_light" mapped to "object"
    10: 6,  # "pole" mapped to "object"
    11: 6,  # "cone" mapped to "object"
    12: 4,  # "bicycle" mapped to "vehicle"
    13: 4,  # "motorcycle" mapped to "vehicle"
    14: 3,  # "building" mapped to "structure"
    15: 2,  # "vegetation" mapped to "vegetation"
    16: 2,  # "tree_trunk" mapped to "vegetation"
    17: 0,  # "curb" mapped to "ground"
    18: 1,  # "road" mapped to "road"
    19: 1,  # "lane_marker" mapped to "road"
    20: 0,  # "other_ground" mapped to "ground"
    21: 0,  # "walkable" mapped to "ground"
    22: 0,  # "sidewalk" mapped to "ground"
}
waymo_metainfo = dict(
    classes=class_names, seg_label_mapping=waymo_labels_map, max_label=22)

# define semantic kitti config
kitti_data_root = '/mnt/d/semantickitti/'
kitti_labels_map = {
    0: 8,  # "unlabeled" mapped to "unlabled"
    1: 7,  # "outlier" mapped to "outliers"
    10: 4,  # "car" mapped to "vehicle"
    11: 4,  # "bicycle" mapped to "vehicle"
    13: 4,  # "bus" mapped to "vehicle"
    15: 4,  # "motorcycle" mapped to "vehicle"
    16: 4,  # "on-rails" mapped to "vehicle"
    18: 4,  # "truck" mapped to "vehicle"
    20: 4,  # "other-vehicle" mapped to "vehicle"
    30: 5,  # "person" mapped to "humans"
    31: 5,  # "bicyclist" mapped to "humans"
    32: 5,  # "motorcyclist" mapped to "humans"
    40: 1,  # "road" mapped to "road"
    44: 1,  # "parking" mapped to "road"
    48: 0,  # "sidewalk" mapped to "ground"
    49: 0,  # "other-ground" mapped to "ground"
    50: 3,  # "building" mapped to "structure"
    51: 6,  # "fence" mapped to "object"
    52: 3,  # "other-structure"  mapped to "structure"
    60: 1,  # "lane-marking" mapped to "road"
    70: 2,  # "vegetation" mapped to "vegetation"
    71: 2,  # "trunk" mapped to "vegetation"
    72: 0,  # "terrain" mapped to "ground"
    80: 6,  # "pole" mapped to "object"
    81: 6,  # "traffic-sign" mapped to "object"
    99: 6,  # "other-object" mapped to "object"
    252: 4, # "moving-car" mapped to "vehicle"
    253: 5, # "moving-bicyclist" mapped to "humans"
    254: 5, # "moving-person" mapped to "humans"
    255: 5, # "moving-motorcyclist" mapped to "humans"
    256: 4, # "moving-on-rails" mapped to "vehicle"
    257: 4, # "moving-bus" mapped to "vehicle"
    258: 4, # "moving-truck" mapped to "vehicle"
    259: 4  # "moving-other-vehicle" mapped to "vehicle"
}
kitti_metainfo = dict(
    classes=class_names, seg_label_mapping=kitti_labels_map, max_label=259)

# define target config
target_data_root = '/mnt/d/target/kitti_format_v1/'
target_labels_map = {
    2: 0,  # "ground" mapped to "ground"
    4: 2,  # "vegetation" mapped to "vegetation"
    6: 3,  # "structure" mapped to "structure"
    7: 7,  # "outlier" mapped to "outlier"
    11: 1,  # "road" mapped to "road"
    64: 4,  # "vehicle" mapped to "vehicle"
    65: 5,  # "humans" mapped to "humans"
    66: 6   # "object" mapped to "object"
}
target_metainfo = dict(
    classes=class_names, seg_label_mapping=target_labels_map, max_label=66)

fov_settings = {
    "waymo": {"up": 2.49875656, "down": -17.886348},
    "kitti": {"up": 3.0, "down": -25.0},
    "target": {"up": 22.5, "down": -22.5}
}

# Specify data loading an augmentations
backend_args = None

pre_transform = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05])
]

pre_transform_5channel = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05])
]

waymo_train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05]),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.00, 0.05],
        max_sat_range=[0.92, 0.97],
        indices=[3],
        square_comp_indices=[3]),
    dict(
        type='FeatureDropout',
        indices=[3],
        prob=0.2),
    dict(
        type='RoadJiggle',
        instance_classes=[0, 1],
        jiggle_range=[-0.025, 0.025],
        prob=0.5),
    dict(
        type='FrustumMix',
        H=64,
        W=512,
        fov_up=fov_settings["waymo"]["up"],
        fov_down=fov_settings["waymo"]["down"],
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='InstanceCopy',
        instance_classes=[5, 6],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["waymo"]["up"],
        fov_down=fov_settings["waymo"]["down"],
        ignore_index=8),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
waymo_test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.02],
        max_sat_range=[0.95],
        indices=[3],
        square_comp_indices=[3]),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["waymo"]["up"],
        fov_down=fov_settings["waymo"]["down"],
        ignore_index=8),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['num_points'])
]
waymo_tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.02],
        max_sat_range=[0.95],
        indices=[3],
        square_comp_indices=[3]),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["waymo"]["up"],
        fov_down=fov_settings["waymo"]["down"],
        ignore_index=8),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[-3.1415926, 3.1415926],
                            scale_ratio_range=[0.95, 1.05],
                            translation_std=[0.05, 0.05, 0.05])
                    ],
                    [
                        dict(
                            type='Pack3DDetInputs',
                            keys=['points'],
                            meta_keys=['num_points'])
                    ]])
]

kitti_train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05]),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.00, 0.05],
        max_sat_range=[0.92, 0.97],
        indices=[3],
        square_comp_indices=[3]),
    dict(
        type='FeatureDropout',
        indices=[3],
        prob=0.2),
    dict(
        type='RoadJiggle',
        instance_classes=[0, 1],
        jiggle_range=[-0.025, 0.025],
        prob=0.5),
    dict(
        type='FrustumMix',
        H=64,
        W=512,
        fov_up=fov_settings["kitti"]["up"],
        fov_down=fov_settings["kitti"]["down"],
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='InstanceCopy',
        instance_classes=[5, 6],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["kitti"]["up"],
        fov_down=fov_settings["kitti"]["down"],
        ignore_index=8),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
kitti_test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.02],
        max_sat_range=[0.95],
        indices=[3],
        square_comp_indices=[3]),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["kitti"]["up"],
        fov_down=fov_settings["kitti"]["down"],
        ignore_index=8),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['num_points'])
]
kitti_tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.02],
        max_sat_range=[0.95],
        indices=[3],
        square_comp_indices=[3]),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["kitti"]["up"],
        fov_down=fov_settings["kitti"]["down"],
        ignore_index=8),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[-3.1415926, 3.1415926],
                            scale_ratio_range=[0.95, 1.05],
                            translation_std=[0.05, 0.05, 0.05])
                    ],
                    [
                        dict(
                            type='Pack3DDetInputs',
                            keys=['points'],
                            meta_keys=['num_points'])
                    ]])
]

target_train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05]),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.00, 0.05],
        max_sat_range=[0.92, 0.97],
        indices=[3, 4],
        square_comp_indices=[3]),
    dict(
        type='FeatureDropout',
        indices=[3, 4],
        prob=0.2),
    dict(
        type='RoadJiggle',
        instance_classes=[0, 1],
        jiggle_range=[-0.025, 0.025],
        prob=0.5),
    dict(
        type='FrustumMix',
        H=64,
        W=512,
        fov_up=fov_settings["target"]["up"],
        fov_down=fov_settings["target"]["down"],
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform_5channel,
        prob=1.0),
    dict(
        type='InstanceCopy',
        instance_classes=[5, 6],
        pre_transform=pre_transform_5channel,
        prob=1.0),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["target"]["up"],
        fov_down=fov_settings["target"]["down"],
        ignore_index=8),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
target_test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.02],
        max_sat_range=[0.95],
        indices=[3, 4],
        square_comp_indices=[3]),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["target"]["up"],
        fov_down=fov_settings["target"]["down"],
        ignore_index=8),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['num_points'])
]
target_tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='HistogramEqualization',
        min_sat_range=[0.02],
        max_sat_range=[0.95],
        indices=[3, 4],
        square_comp_indices=[3]),
    dict(
        type='RangeInterpolation',
        H=64,
        W=1024,
        fov_up=fov_settings["target"]["up"],
        fov_down=fov_settings["target"]["down"],
        ignore_index=8),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[-3.1415926, 3.1415926],
                            scale_ratio_range=[0.95, 1.05],
                            translation_std=[0.05, 0.05, 0.05])
                    ],
                    [
                        dict(
                            type='Pack3DDetInputs',
                            keys=['points'],
                            meta_keys=['num_points'])
                    ]])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        datasets= {
            "target" : dict(
                type="SemKittiReducedDataset",
                data_root=target_data_root,
                ann_file='target_infos_train.pkl',
                pipeline=target_train_pipeline,
                metainfo=target_metainfo,
                modality=input_modality,
                ignore_index=8,
                backend_args=backend_args
            ),
            "kitti" : dict(
                type="SemKittiReducedDataset",
                data_root=kitti_data_root,
                ann_file='semantickitti_infos_train.pkl',
                pipeline=kitti_train_pipeline,
                metainfo=kitti_metainfo,
                modality=input_modality,
                ignore_index=8,
                backend_args=backend_args
            ),
            "waymo" : dict(
                type="SemKittiReducedDataset",
                data_root=waymo_data_root,
                ann_file='waymo_infos_train.pkl',
                pipeline=waymo_train_pipeline,
                metainfo=waymo_metainfo,
                modality=input_modality,
                ignore_index=8,
                backend_args=backend_args
            )
        }))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        datasets= {
            "target" : dict(
                type="SemKittiReducedDataset",
                data_root=target_data_root,
                ann_file='target_infos_val.pkl',
                pipeline=target_test_pipeline,
                metainfo=target_metainfo,
                modality=input_modality,
                ignore_index=8,
                test_mode=True,
                backend_args=backend_args
            ),
            "kitti" : dict(
                type="SemKittiReducedDataset",
                data_root=kitti_data_root,
                ann_file='semantickitti_infos_val.pkl',
                pipeline=kitti_test_pipeline,
                metainfo=kitti_metainfo,
                modality=input_modality,
                ignore_index=8,
                test_mode=True,
                backend_args=backend_args
            ),
            "waymo" : dict(
                type="SemKittiReducedDataset",
                data_root=waymo_data_root,
                ann_file='waymo_infos_val.pkl',
                pipeline=waymo_test_pipeline,
                metainfo=waymo_metainfo,
                modality=input_modality,
                ignore_index=8,
                test_mode=True,
                backend_args=backend_args
            )
        }))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiSegMetric',
    log_path='work_dirs/frnet-mixed_seg',
    ignore_index=8)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')
