import numpy as np
class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'certain' : 4
               }

test_class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'certain' : 4

}

img_size = (128, 128, 3)
NUM_CLASSES = 5
dataset_path = 'data/dataset/labelled-pool/DataODAEL/train_complex'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
return_index = True
return_index = True
train_mode = 'classification'