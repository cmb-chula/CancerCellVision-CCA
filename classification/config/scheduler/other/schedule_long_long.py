def schedule(step, optimizer):
    init_lr = initial_lr
    if(step < warmup_step):
        init_lr = initial_lr * step / warmup_step
    for i in STEP:
        if(step > i):
            init_lr *= 0.1
    optimizer.lr.assign(init_lr)

NUM_ITERATION = 48000
STEP = [32000, 44000]
val_freq = 2000
initial_lr = 5e-4
warmup_step = 1000