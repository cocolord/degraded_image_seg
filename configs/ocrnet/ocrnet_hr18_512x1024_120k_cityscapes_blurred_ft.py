_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes_blurred.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
resume_from = 'work_dirs/ocrnet_hr18_512x1024_80k_cityscapes_20200614_230521-c2e1dd4a.pth'

runner = dict(type='IterBasedRunner', max_iters=120000)
data = dict(
    samples_per_gpu=3)