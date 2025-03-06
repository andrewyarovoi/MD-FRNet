_base_ = [
    '../_base_/datasets/mixed_seg.py', '../_base_/models/frnet_ppt.py',
    '../_base_/schedules/onecycle-100k.py', '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['frnet.datasets', 'frnet.datasets.transforms', 'frnet.models', 'frnet.metrics'],
    allow_failed_imports=False)

fov_settings = {
    "waymo": {"up": 2.49875656, "down": -17.886348},
    "kitti": {"up": 3.0, "down": -25.0},
    "target": {"up": 22.5, "down": -22.5}
}

model = dict(
    data_preprocessor=dict(
        H=64, W=512, fov_settings=fov_settings, num_contexts=3, ignore_index=8),
    backbone=dict(output_shape=(64, 512)),
    decode_head=dict(num_classes=9, ignore_index=8),
    auxiliary_head=[
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=9,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=8),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=9,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=8,
            indices=2),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=9,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=8,
            indices=3),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=9,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=8,
            indices=4),
    ])
