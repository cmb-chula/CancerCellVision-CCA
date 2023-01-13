import numpy as np
class_mapper = {
                'R'   :   0,        
                'G'   :   1,
               }

test_class_mapper = {
                'R'   :   0,        
                'G'   :   1,  
                'U'   :   2,          
        
}

img_size = (128, 128, 3)
NUM_CLASSES = 3
# train_path = 'data/dataset/labelled-pool/CTC_entering2/train/brightfield'
# val_path = 'data/dataset/labelled-pool/CTC_entering2/val/brightfield'
# test_path = 'data/dataset/labelled-pool/CTC_entering2/val/brightfield'
train_path = 'data/dataset/labelled-pool/CTC/train_noU/brightfield'
val_path = 'data/dataset/labelled-pool/CTC/val_noU/brightfield'
test_path = 'data/dataset/labelled-pool/CTC/train/brightfield'


BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
return_index = False
train_mode = 'classification'
resize_mode = 'pad'
# backbone = 'effnet'
backbone = 'convnext'
pseudo = False

read_U = True
read_RGU = False
visualize = False
quality = False
