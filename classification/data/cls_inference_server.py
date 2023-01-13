import threading
import time
import numpy as np
from collections import deque


class ClassificationInferenceServer(threading.Thread):
    def __init__(self, model, batch_size = 32, limit=False, max_queue_size=30, cls_weight = 1, seek_data = False,  target=None, name=None):
        # super(DummyInferenceEngine, self).__init__()
        threading.Thread.__init__(self)
        self.target = target
        self.name = name
        self.model = model
        self.limit = limit
        self.max_queue_size = max_queue_size
        self.buffer = deque(maxlen=max_queue_size)
        self.prediction_result = []
        self.can_query_result = False
        self.cls_weight = cls_weight
        self.batch_size = batch_size
        self.batch_buffer = []
        self.loc_buffer = []
        self.seek_data = seek_data  

    def push(self, X):
        """Push image into buffer (shared memory).
        # Arguments
            X: Arrray(H x W x 3), body image.
        # Returns
            None
        """
        # print(len(self.buffer))
        while(len(self.buffer) == self.max_queue_size):
            time.sleep(1e-6)
        self.buffer.append(X)

    def grab(self):
        """Grab image from the buffer (shared memory).
        # Arguments
            engine : InferenceEngine, engine used for inference
        # Returns
            X: Arrray(H x W x 3), body image.
        """

        while(len(self.buffer) == 0):
            time.sleep(1e-6)
        X = self.buffer.popleft()
        return X

    def get_prediction_result(self):
        while(self.can_query_result == False):
            time.sleep(1e-6)
        output = self.prediction_result.copy()
        #reset state
        self.can_query_result = False
        self.prediction_result = []
        return output

    def inference_batch(self):
        x = np.array(self.batch_buffer, dtype = np.float32)
        res = self.model(x)
        for idx, (coors, cls_conf) in enumerate(zip(np.array(self.loc_buffer), res.numpy())):

            xmin, ymin, xmax, ymax, conf = coors
            # if(x[idx].mean() > 240): conf = 0
            cls_conf  = cls_conf[0]
            rescore_conf = (self.cls_weight) * cls_conf + (1 - self.cls_weight) * conf
            if(not  self.seek_data):
                self.prediction_result.append([xmin, ymin, xmax, ymax, rescore_conf])
                # self.prediction_result.append([xmin, ymin, xmax, ymax, conf, rescore_conf])

            else:
                self.prediction_result.append([xmin, ymin, xmax, ymax, (xmin, ymin, xmax, ymax, conf), rescore_conf,])


            
        self.batch_buffer = []
        self.loc_buffer = []

    # def inference_batch(self):
    #     x = np.array(self.batch_buffer, dtype = np.float32)
    #     res = self.model(x)
    #     for idx, (coors, embedding) in enumerate(zip(np.array(self.loc_buffer), res.numpy())):
    #         xmin, ymin, xmax, ymax, conf = coors
    #         self.prediction_result.append([xmin, ymin, xmax, ymax, np.array(embedding, dtype = np.float16)])
    #     self.batch_buffer = []
    #     self.loc_buffer = []

    def run(self):
        """Start image loading thread.
        """
        while(True):
            x = self.grab()
            if(x is None): break
            xmin, ymin, xmax, ymax, conf, img = x
            # if(conf < 0.1 or conf > 0.9):
            #     self.prediction_result.append([xmin, ymin, xmax, ymax, conf ])
            #     continue
            self.batch_buffer.append(img)
            self.loc_buffer.append([xmin, ymin, xmax, ymax, conf])
            if(len(self.batch_buffer) == self.batch_size): 
                self.inference_batch()
        if(len(self.batch_buffer) > 0): self.inference_batch()
        self.prediction_result = self.prediction_result
        self.can_query_result = True


    def run_single_instance(self):
        """Start image loading thread.
        """
        while(True):
            x = self.grab()
            if(x is None): break
            xmin, ymin, xmax, ymax, conf, img = x
            # if(conf < 0.1 or conf > 0.9):
            #     self.prediction_result.append([xmin, ymin, xmax, ymax, conf ])
            #     continue
            res = self.model(np.array(img[None, ], dtype = np.float32))
            rescore_conf = (self.cls_weight) * res[0][0].numpy() + (1 - self.cls_weight) * conf
            self.prediction_result.append([xmin, ymin, xmax, ymax, rescore_conf])
        self.prediction_result = np.array(self.prediction_result, dtype = np.float32)
        self.can_query_result = True
