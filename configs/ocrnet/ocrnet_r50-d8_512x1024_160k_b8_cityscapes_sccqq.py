_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', '../_base_/datasets/cityscapes_sccqq.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(pretrained='torchvision://resnet50', backbone=dict(depth=50))
optimizer = dict(lr=0.02)
lr_config = dict(min_lr=2e-4)
