import numpy as np
class_mapper = {
                'positive'       :   0,        
                'negative'   :   1,          
               }

test_class_mapper = {
                'positive'       :   0,        
                'negative'   :   1,          

}

img_size = (512, 512, 3)
NUM_CLASSES = 2
dataset_path = 'data/dataset/labelled-pool/COVID/new'
BATCH_SIZE = 8
return_embedding = False
pretrain_path = None
return_index = False
train_mode = 'classification'
resize_mode = 'pad'