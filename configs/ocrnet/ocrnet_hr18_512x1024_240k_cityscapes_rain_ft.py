_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes_rain.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
resume_from = 'work_dirs/ocrnet_hr18_512x1024_240k_cityscapes_foggy_ft/iter_228000.pth'

runner = dict(type='IterBasedRunner', max_iters=240000)
data = dict(
    samples_per_gpu=3)