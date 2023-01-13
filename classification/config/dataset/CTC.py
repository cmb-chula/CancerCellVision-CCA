import numpy as np
class_mapper = {
                'R'   :   0,        
                'G'   :   1,          
                'U'   :   1,                     
               }

test_class_mapper = {
                'R'   :   0,        
                'G'   :   1,          
                'U'   :   1,          
}

img_size = (160, 160, 3)
NUM_CLASSES = 2
train_path = 'data/dataset/labelled-pool/CTC_entering2/train/brightfield'
val_path = 'data/dataset/labelled-pool/CTC_entering2/val/brightfield'
test_path = 'data/dataset/labelled-pool/CTC_entering2/val/brightfield'

read_U = False
read_RGU = False

BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
return_index = False
train_mode = 'classification'
resize_mode = 'pad'
backbone = 'convnext'
visualize = False
quality = False
pseudo = False
