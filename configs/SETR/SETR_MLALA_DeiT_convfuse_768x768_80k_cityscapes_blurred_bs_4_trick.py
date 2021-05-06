_base_ = [
    '../_base_/models/setr_mlala_convfuse.py',
    '../_base_/datasets/cityscapes_768x768_blurred.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    backbone=dict(img_size=768,pos_embed_interp=True, drop_rate=0.,mla_channels=256,
                  model_name='deit_base_distilled_path16_384', mla_index=(2,5,8,11), embed_dim=768, depth=12, num_heads=12),
    decode_head=dict(img_size=768,mla_channels=256,mlahead_channels=128,num_classes=19,
        # 官方文档参数
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000), 
    ),
    auxiliary_head=[
        dict(
        type='VIT_MLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=0,
        img_size=768,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VIT_MLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=1,
        img_size=768,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
            # DeepLabV3 权重
            class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
            )),
        dict(
        type='VIT_MLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=2,
        img_size=768,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VIT_MLA_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=3,
        img_size=768,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        ])

optimizer = dict(lr=0.002, weight_decay=0.0,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=2.)})
)

crop_size = (768, 768)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True
data = dict(samples_per_gpu=2)
