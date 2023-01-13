multiplier = 1

def schedule(step, optimizer):
    init_lr = initial_lr
    if(step < warmup_step):
        init_lr = initial_lr * step / warmup_step
    for i in STEP:
        if(step > i):
            init_lr *= 0.1
            
    optimizer.lr.assign(init_lr)


NUM_ITERATION = int(24000 *multiplier)
STEP = [int(15000*multiplier), int(21000*multiplier)]
val_freq = 1000
initial_lr = 5e-4
warmup_step = 500
# NUM_ITERATION = int(30000 *multiplier)
# STEP = [int(22500*multiplier), int(27000*multiplier)]
# val_freq = 1000
# initial_lr = 5e-4
# warmup_step = 500