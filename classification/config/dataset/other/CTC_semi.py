import numpy as np
class_mapper = {
                'R'       :   0,        
                'G'   :   1,          
                'U'   :   1,          

               }

test_class_mapper = {
                'R'   :   0,        
                'G'   :   1,          
                'U'   :   1,          
}

img_size = (64, 64, 3)
NUM_CLASSES = 2
dataset_path = 'data/dataset/labelled-pool/CTC_semi_no_processing/brightfield'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
return_index = False
train_mode = 'classification'
resize_mode = 'pad'
backbone = 'effnet'