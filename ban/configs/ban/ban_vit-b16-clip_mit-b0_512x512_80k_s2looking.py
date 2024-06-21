_base_ = [
    '../_base_/models/ban_vit-b16.py', 
    '../common/standard_512x512_40k_s2looking.py']

crop_size = (512, 512)
checkpoint_file = 'pretrain/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    asymetric_input=True,
    encoder_resolution=dict(
        size=(224, 224),
        mode='bilinear'),
    image_encoder=dict(
        frozen_exclude=[]),
    decode_head=dict(
        type='BitemporalAdapterHead',
        ban_cfg=dict(
            clip_channels=768,
            fusion_index=[1, 2, 3],
            side_enc_cfg=dict(
                type='mmseg.MixVisionTransformer',
                init_cfg=dict(
                    type='Pretrained', checkpoint='pretrain/mit_b0_20220624-7e0fe6dd.pth'),
                in_channels=3,
                embed_dims=32,
                num_stages=4,
                num_layers=[2, 2, 2, 2],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1)),
        ban_dec_cfg=dict(
            type='BAN_MLPDecoder',
            in_channels=[32, 64, 160, 256],
            channels=128,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.),
            'mask_decoder': dict(lr_mult=10.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=8000))

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=1)