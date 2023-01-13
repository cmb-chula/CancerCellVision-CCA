import os



def build_model_torch(cfg):
    assert cfg.backbone is not None

    if(cfg.train_mode == 'classification'):
        from .model_torch import ClassificationModel
        model = ClassificationModel(input_shape = cfg.img_size, backbone = cfg.backbone, num_classes = cfg.NUM_CLASSES)
    elif (cfg.train_mode == 'CTC'):
        from .segmentation_model import SegmentationModel
        model = SegmentationModel(input_shape = cfg.img_size, backbone = cfg.backbone, num_classes = cfg.NUM_CLASSES)
    # if(cfg.pretrain_path != None):
    #     model.load_weights(cfg.pretrain_path, by_name = True, skip_mismatch = True)
    return model

def build_model(cfg):
    import tensorflow as tf
    from .model import get_backbone
    from .metric import macrof1

    backbone = get_backbone(input_shape = cfg.img_size, backbone_nme = cfg.backbone)
    print(cfg.backbone)
    assert backbone is not None

    if(cfg.train_mode == 'classification'):
        from .model import get_classification_model
        model = get_classification_model(backbone, input_shape = cfg.img_size, num_classes = cfg.NUM_CLASSES, return_embedding=cfg.return_embedding, visualize = cfg.visualize)
    elif (cfg.train_mode == 'relocation'):
        from .model import get_relocation_model
        model = get_relocation_model(backbone, input_shape = cfg.img_size, num_classes = cfg.NUM_CLASSES, return_embedding=cfg.return_embedding)
    elif (cfg.train_mode == 'relocation_aux'):
        from .model import get_relocation_aux_model
        model = get_relocation_aux_model(backbone, input_shape = cfg.img_size, num_classes = cfg.NUM_CLASSES, return_embedding=cfg.return_embedding)
    elif (cfg.train_mode == 'CTC'):
        from .model import get_segmentation_model
        model = get_segmentation_model(backbone, input_shape = cfg.img_size, num_classes = cfg.NUM_CLASSES, return_embedding=cfg.return_embedding)

    if(cfg.pretrain_path != None):
        model.load_weights(cfg.pretrain_path, by_name = True, skip_mismatch = True)
    return model