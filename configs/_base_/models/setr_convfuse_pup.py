# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VIT_ConvFuse',
        model_name='deit_base_distilled_path16_384',
        img_size=768,
        patch_size=16, 
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16, 
        num_classes=19,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        conv_type='unet'
        ),
    decode_head=dict(
        type='VisionTransformerUpHeadConvFuse',
        in_channels=1024,
        channels=512,
        in_index=23,
        img_size=768,
        embed_dim=1024,
        num_classes=19,
        norm_cfg=norm_cfg,
         num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')

