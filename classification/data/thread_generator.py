import threading
import time
import numpy as np
from collections import deque

class ThreadGenerator(threading.Thread):
    def __init__(self, generator, max_queue_size = 10):
        # super(DummyInferenceEngine, self).__init__()
        threading.Thread.__init__(self)
        self.generator = generator
        self.buffer = deque(maxlen=max_queue_size)
        self.max_queue_size = max_queue_size
    def push(self, X):
        while(len(self.buffer) == self.max_queue_size):
            time.sleep(1e-6)
        self.buffer.append(X)

    def grab(self):
        while(len(self.buffer) <= 0):
            time.sleep(1e-6)
        data = self.buffer.popleft()
        return data

    def run(self):
        while(True):
            data = next(self.generator)
            self.push(data)