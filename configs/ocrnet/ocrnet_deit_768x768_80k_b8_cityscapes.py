_base_ = [
    '../_base_/datasets/cityscapes_768x768.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='VIT_MLA',
        model_name='deit_base_distilled_path16_384',
        img_size=768,
        patch_size=16, 
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12, 
        num_classes=19,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        mla_channels=768,
        mla_index=(2,5,8,11)
        ),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=768,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=768,
            in_index=3,
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
)
optimizer = dict(lr=0.002, weight_decay=0.0,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=5.)})
)
crop_size = (768, 768)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True
data = dict(samples_per_gpu=2)