from .thread_generator import ThreadGenerator

def make_data_loader(cfg, files, is_train=False, fetch_into_mem = False, get_label = True, limit = None):
    from .train_loader import TrainLoader
    loader = TrainLoader(target_size=(cfg.img_size[:2])).run(cfg, files, cfg.BATCH_SIZE, 
                        training=is_train, augment=is_train, get_label = get_label)
    if(not fetch_into_mem):
        loader = ThreadGenerator(loader)
        loader.setDaemon(True)
        loader.start()
    else:
        X_val, Y_val = TrainLoader.fetch_data_into_mem(loader, limit = limit)
        loader = (X_val, Y_val)

    return loader


def make_CTC_loader(cfg, files, is_train=False, fetch_into_mem = False, get_label = True, limit = None):
    from .CTC_loader import CTCLoader
    loader = CTCLoader(target_size=(cfg.img_size[:2]), resize_mode = 'pad').run(cfg, files, cfg.BATCH_SIZE, 
                        training=is_train, augment=is_train, get_label = get_label)
    if(not fetch_into_mem):
        loader = ThreadGenerator(loader)
        loader.setDaemon(True)
        loader.start()
    else:
        X_val, Y_val = CTCLoader.fetch_data_into_mem(loader, limit = limit)
        loader = (X_val, Y_val)
    return loader


def make_CTC_semantic_loader(cfg, files, is_train=False, fetch_into_mem = False, get_label = True, limit = None):
    from .CTC_semantic_loader import CTCSemanticLoader
    loader = CTCSemanticLoader(target_size=(cfg.img_size[:2]), resize_mode = 'pad').run(cfg, files, cfg.BATCH_SIZE, 
                        training=is_train, augment=is_train, get_label = get_label)
    if(not fetch_into_mem):
        loader = ThreadGenerator(loader)
        loader.setDaemon(True)
        loader.start()
    else:
        X_val, Y_val = CTCLoader.fetch_data_into_mem(loader, limit = limit)
        loader = (X_val, Y_val)
    return loader


def make_unsup_loader(base_path, data_path, obj_loc, batch_size = 128, is_train=False, get_label = True, limit = None, model = None):
    from .unsup_loader import UnsuperviseLoader 
    loader = UnsuperviseLoader(obj_loc).run(base_path, data_path, batch_size, training=is_train, augment=is_train, get_label = get_label)

    loader = ThreadGenerator(loader)
    loader.setDaemon(True)
    loader.start()
    
    return loader
