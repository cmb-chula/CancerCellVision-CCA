def schedule(step, optimizer):
    init_lr = initial_lr
    if(step < warmup_step):
        init_lr = initial_lr * step / warmup_step
    else:
        init_lr = initial_lr * (NUM_ITERATION - step) / warmup_step
    optimizer.lr.assign(init_lr)
multiplier = 1
NUM_ITERATION = 10000 * multiplier
val_freq = 500
initial_lr = 5e-4
warmup_step = 5000 * multiplier
