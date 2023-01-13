import numpy as np
test_class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    1,
                'Tumorcells' :      1
               }

class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    1,
                'Tumorcells' :      1
}

img_size = (128, 128, 3)
NUM_CLASSES = 4
dataset_path = 'data/dataset/labelled-pool/DataODAEL_extra/train'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None