_base_ = [
    '../_base_/models/setr_mlala_convfuse_unetlike_contrast.py',
    '../_base_/datasets/cityscapes_768x768_paired_foggy.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

model = dict(
    backbone=dict(img_size=768,pos_embed_interp=True, drop_rate=0.,mla_channels=256,
                  model_name='deit_base_distilled_path16_384', mla_index=(2,5,8,11), embed_dim=768, depth=12, num_heads=12),
    decode_head=dict(img_size=768,mla_channels=256,mlahead_channels=128,num_classes=19,
        # 官方文档参数
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000), 
    ),
   )

optimizer = dict(lr=0.001, weight_decay=0.0,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=1.)})
)

crop_size = (768, 768)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True
data = dict(samples_per_gpu=1)
