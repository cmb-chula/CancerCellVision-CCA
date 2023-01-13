# # optimizer
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     # step=[ 3, 5])
#     step=[ 5, 7])

# total_epochs = 8


# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    # step=[ 3, 5])
    step=[ 10, 14])

total_epochs = 16

