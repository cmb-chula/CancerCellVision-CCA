def schedule(step, optimizer):
    init_lr = initial_lr
    if(step < warmup_step):
        init_lr = initial_lr * step / warmup_step
    for i in STEP:
        if(step > i):
            init_lr *= 0.1
    optimizer.lr.assign(init_lr)

NUM_ITERATION = 6000
STEP = [4000, 5500]
val_freq = 500
initial_lr = 1e-3
warmup_step = 500