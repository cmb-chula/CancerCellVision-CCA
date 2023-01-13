import threading
import time
import numpy as np
from collections import deque
from .base_loader import BaseLoader


class InferenceEngine(threading.Thread, BaseLoader):
    def __init__(self, file_list, batch_size=64, limit=False, max_queue_size=10, target_size=(240, 96),  target=None, name=None):
        # super(DummyInferenceEngine, self).__init__()
        threading.Thread.__init__(self)
        BaseEngine.__init__(self, target_size)
        self.target = target
        self.name = name
        self.file_list = file_list
        self.batch_size = batch_size
        self.limit = limit
        self.max_queue_size = max_queue_size
        self.buffer = deque(maxlen=max_queue_size)

    def push(self, X):
        """Push image into buffer (shared memory).
        # Arguments
            X: Arrray(H x W x 3), body image.
        # Returns
            None
        """
        while(len(self.buffer) == self.max_queue_size):
            time.sleep(1e-6)
        self.buffer.append(X)

    @staticmethod
    def grab(engine):
        """Grab image from the buffer (shared memory).
        # Arguments
            engine : InferenceEngine, engine used for inference
        # Returns
            X: Arrray(H x W x 3), body image.
        """

        while(len(engine.buffer) <= 0):
            time.sleep(1e-6)
        X = engine.buffer.popleft()
        return X

    def run(self):
        """Start image loading thread.
        """

        from tqdm import tqdm
        batches = self.make_batches(len(self.file_list), self.batch_size)
        for batch_index, (batch_start, batch_end) in tqdm(enumerate(batches)):
            if(batch_index * self.batch_size > self.limit and self.limit):
                break
            batch_ids = self.file_list[batch_start:batch_end]
            from multiprocessing.dummy import Pool
            p = Pool(4)
            X = np.array(p.map(self.load_image, [x for x in batch_ids]))
            p.close()
            self.push(X)
        self.push(None)
        return
