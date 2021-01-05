_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes_blurred.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))
resume_from = 'work_dirs/ocrnet_hr18s_512x1024_80k_cityscapes_20200601_222735-55979e63.pth'

runner = dict(type='IterBasedRunner', max_iters=120000)
data = dict(
    samples_per_gpu=8)