NUM_ITERATION = 20000
STEP = [10000, 18000]
val_freq = 1000
initial_lr = 1e-3
warmup_step = 500
def schedule(step):
    init_lr = initial_lr
    if(step < warmup_step):
        init_lr = initial_lr * step / warmup_step
    for i in STEP:
        if(step > i):
            init_lr *= 0.1
    return init_lr 
