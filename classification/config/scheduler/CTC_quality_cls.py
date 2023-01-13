def schedule(step, optimizer):
    init_lr = initial_lr
    if(step < warmup_step):
        init_lr = initial_lr * step / warmup_step
    for i in STEP:
        if(step > i):
            init_lr *= 0.1
    optimizer.lr.assign(init_lr)
multiplier = 1#2
NUM_ITERATION = 5000# * multiplier
STEP = [2000, 6000]
val_freq = 500
initial_lr = 1e-4
warmup_step = 500 * multiplier

# multiplier = 1#2
# NUM_ITERATION = 500000# * multiplier
# STEP = [250000, 600000]
# val_freq = 500
# initial_lr = 1e-4
# warmup_step = 500 * multiplier
