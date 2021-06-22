_base_ = [
    '../_base_/models/setr_convfuse_pup.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(img_size=512,align_corners=False, pos_embed_interp=True,drop_rate=0.,num_classes=150,embed_dim=768, depth=12, num_heads=12,conv_type='single'),
    decode_head=dict(img_size=512,align_corners=False,num_conv=4,upsampling_method='bilinear',embed_dim=768, in_index=11,
    num_upsampe_layer=4,num_classes=150),
    )

optimizer = dict(lr=0.001, weight_decay=0.0,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=10.)})
)

crop_size = (512, 512)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(384, 384))
find_unused_parameters = True
data = dict(samples_per_gpu=6)
