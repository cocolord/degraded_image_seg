_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', '../_base_/datasets/cityscapes_foggy.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(pretrained='torchvision://resnet50', backbone=dict(depth=50))
optimizer = dict(lr=0.001)
lr_config = dict(min_lr=2e-4)
resume_from = 'work_dirs/ocrnet_r50-d8_512x1024_160k_b8_cityscapes_blurred/iter_160000.pth'
runner = dict(type='IterBasedRunner', max_iters=200000)