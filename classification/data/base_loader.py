import time
import numpy as np
from PIL import Image, ImageOps
import cv2


class BaseLoader():
    def __init__(self, target_size=(240, 96), resize_mode = 'default'):
        self.resize_mode = resize_mode
        self.target_size = target_size
    def make_batches(self, size, batch_size):
        """Returns a list of batch indices (tuples of indices).
        # Arguments
            size: Integer, total size of the data to slice into batches.
            batch_size: Integer, batch size.
        # Returns
            A list of tuples of array indices.
        """
        num_batches = (size + batch_size - 1) // batch_size  # round up
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(num_batches)]

    
    def load_image(self, img_path):
        """Returns an image give a file path.
        # Arguments
            img_path: String, path of an image.
            target_size: (Int, Int), the final size of the image (H, W).
        # Returns
            Array(H x W x 3) an read image from the given datapath. The image is in a RGB format.
        """
        image = Image.open(img_path)
        # if(self.resize_mode == 'pad'):
        #     image = ImageOps.pad(image, (self.target_size[0], self.target_size[1]), color='white')
        # print(image.shape)
        # image = np.array(cv2.resize(image, (self.target_size[0], self.target_size[1])), dtype = np.float32)#np.array(cv2.resize(image, (self.target_size[0], self.target_size[1]) ) , dtype=np.float32)
        # kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        # image = cv2.filter2D(image, -1, kernel)
        image = np.array(image, dtype = np.float32)
        return image