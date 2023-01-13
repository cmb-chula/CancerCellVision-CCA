def lr_schedule(epoch, lr):
    init_lr = 1e-3
    if(epoch >= 40): init_lr *= 0.1
    return init_lr

TOTAL_EPOCH  = 55
STEP_PER_EPOCH = 1000