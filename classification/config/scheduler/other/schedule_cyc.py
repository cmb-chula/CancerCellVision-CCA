def schedule(step, optimizer):
    init_lr = initial_lr
    if(step < STEP[0]):
        init_lr = initial_lr * step / STEP[0]
    else:
        if(step < STEP[1]):
            init_lr = initial_lr *  (1 - ((step - STEP[0]) / (STEP[2] - STEP[0])))
        else:
            pass
    if(step < warmup_step):
        init_lr = initial_lr * step / warmup_step
    for i in STEP:
        if(step > i):
            init_lr *= 0.1
    optimizer.lr.assign(init_lr)

NUM_ITERATION = 24000
STEP = [8000, 22000]
val_freq = 1000
initial_lr = 5e-4
warmup_step = 500