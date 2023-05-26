import numpy as np
class_mapper = {
                'R'   :   0,        
                'G'   :   1,          
                'U'   :   2,                     
               }

test_class_mapper = {
                'R'   :   0,        
                'G'   :   1,          
                'U'   :   2,          
}

img_size = (128, 128, 6)
NUM_CLASSES = 3
# train_path = 'data/dataset/labelled-pool/CTC_entering2/train/brightfield'
# val_path = 'data/dataset/labelled-pool/CTC_entering2/val/brightfield'
# test_path = 'data/dataset/labelled-pool/CTC_entering2/val/brightfield'

train_path = 'data/dataset/labelled-pool/CTC/train/brightfield'
val_path = 'data/dataset/labelled-pool/CTC/val/brightfield'
test_path = 'data/dataset/labelled-pool/CTC/val/brightfield'



BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
return_index = False
train_mode = 'classification'
resize_mode = 'pad'
backbone = 'ConvNext-L'
pseudo = False

FL_only = False
read_U = False
read_RGU = True
visualize = False
quality = False


