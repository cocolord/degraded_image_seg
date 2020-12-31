_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_sccqq.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(
        depth=101,
        norm_cfg=norm_cfg,
        norm_eval=False),
    decode_head=dict(
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
    )
    )
