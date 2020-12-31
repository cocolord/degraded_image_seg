_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes_foggy.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
resume_from = 'work_dirs/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth'

runner = dict(type='IterBasedRunner', max_iters=200000)
data = dict(
    samples_per_gpu=3)