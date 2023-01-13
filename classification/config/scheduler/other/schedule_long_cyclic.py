multiplier = 1

def schedule(step, optimizer):
    one_epoch = 166621 * multiplier / 64
    current_epoch = (step // one_epoch)
    init_lr = initial_lr * ( 10 - int(current_epoch % 10) ) /10 
    optimizer.lr.assign(init_lr)

NUM_ITERATION = int(166621 * 40 * multiplier / 64 )
# STEP = [int(166621 * 10 * (2/3) * multiplier / 64), int(166621 * 10 * (11/12) * multiplier / 64)]
val_freq = 1000
initial_lr = 5e-4
warmup_step = 500