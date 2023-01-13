import numpy as np
test_class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'Other' :      4
               }

class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'Other' :      4
}

img_size = (128, 128, 3)
NUM_CLASSES = 4
dataset_path = '../detection/mmdetection/mmdetection/data/mitotic/WSI/'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
train_mode = 'relocation'
backbone ='effnet'