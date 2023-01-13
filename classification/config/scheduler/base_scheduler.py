def schedule(step, optimizer):
    init_lr = 1e-3
    for i in STEP:
        if(step > i):
            init_lr *= 0.1
    optimizer.lr.assign(init_lr)

NUM_ITERATION = 10000
STEP = [7000, 9000]
val_freq = 500
initial_lr = 1e-3
