import time
import numpy as np
from .base_loader import BaseLoader
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import scipy
from scipy.stats import median_absolute_deviation
import pickle

class CTCLoader(BaseLoader):
    def __init__(self, target_size=(64, 64), resize_mode = 'default'):
        super().__init__(target_size, resize_mode)
        import pickle
        self.augment = False
        self.get_label = False
        self.db = None

    def actual_aug(self, img):
        import cv2
        if(np.random.rand() > 0.5):
            img = img[::-1, :, :]
        if(np.random.rand() > 0.5):
            img = img[:, :: -1, :]
        img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)
        img *= np.random.uniform(0.8, 1.2)
        return img

    def aug(self, X):
        from multiprocessing.dummy import Pool
        p = Pool(8)
        aug_data = p.map(self.actual_aug, X)#[self.actual_aug(x) for x in X]
        p.close()
        return aug_data
    def augment_batch(self, X, Y):
        """Augment training images. This method perform basic transformation and CutMix augmentation
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)
        # Arguments
            X : Array(batch size x H x W x 3), a tensor of body images
            Y : Array(batch size x num_class), a one hot label tensor of body images
        # Returns
            Array(batch size x H x W x 3), body image after augmented.
            Array(batch size x num_class), label after augmented.
        """

        # X = self.aug(X)
        # X = np.array(X)
        # alpha = 0.1
        # Y[Y !=  1] = alpha / (Y.shape[1] - 1)
        # Y[Y == 1] = 1 - alpha
        return X, Y


    def calculate_metric(self, cell, d):
        max_intensity = cell.max() / 255
        cell = 255 * np.array(cell / cell.max(), dtype = np.float32)
        SNR = cell.mean() / cell.std()
        aspect_ratio = max(cell.shape[0] / cell.shape[1], cell.shape[1] / cell.shape[0])
        percentage_maximal = len(cell[cell > 255 * 0.9]) / (cell.shape[0] * cell.shape[1]) 
        percentage_minimal = len(cell[cell > 255 * 0.3]) / (cell.shape[0] * cell.shape[1]) 
        median = np.median(cell) / 255
        mean_thresh = len(cell[cell >  np.mean(cell)])  / (cell.shape[0] * cell.shape[1])
        median_std = median_absolute_deviation(cell.reshape(-1) / 255) + 1e-3
        score = np.sum( np.array(d) * np.log(np.array([ SNR, median, aspect_ratio, median_std, 
                                                    percentage_maximal, percentage_minimal])) )
        return score

    @staticmethod
    def fetch_data_into_mem(generator, limit = None):
        """Load body images and label from the generator into a memory
        # Arguments
            generator : DataEngine, DataEngine generator and its subclass
        # Returns
            Array(nb. data x H x W x 3), all body images yield from the genertor .
            Array(nb. data x num_class), all body images yield from the genertor.
        """

        X, Y = [], []
        while(True):
            try:
                data = next(generator)
                
                x, y = data[0], data[1]
                X += list(x)
                Y += list(y)
                if(limit is not None and len(X) > limit):
                    break
                
            except (StopIteration, TypeError):
                break
        return np.array(X), np.array(Y)

    def query_label(self, file_path, training=False):
        classes = file_path.split('/')[-2]

        one_hot = np.zeros(self.cfg.NUM_CLASSES, dtype = np.float32)
        if(training):
            name = file_path.split('/')[-1].split('.')[0]

            if(self.db is not None and name in self.db):
                conf = self.db[name]
                one_hot[0] = conf
                one_hot[one_hot !=  conf] = (1 - conf) / (one_hot.shape[0] - 1)
            else:
                one_hot[self.cfg.class_mapper[classes]] = 1
                alpha = 0.1
                one_hot[one_hot !=  1] = alpha / (one_hot.shape[0] - 1)
                one_hot[one_hot == 1] = 1 - alpha

        else:
            one_hot[self.cfg.test_class_mapper[classes]] = 1
        return one_hot

    def run(self, cfg, data_path, batch_size=128, training=False, augment=False, get_label = True, return_index = False):

        self.cfg = cfg

        if(cfg.quality):
            self.Z = pickle.load(open('quality_RGU_iter1_0.pkl', "rb" ))
            # self.Z = pickle.load(open('quality_RGU_iter0_1.pkl', "rb" ))


            # self.Z = pickle.load(open('quality_5.pkl', "rb" ))
            # self.mu = np.array([self.Z[i] for i in self.Z]) #.mean()
            # print(self.mu)
        if(cfg.pseudo):
            # self.db = pickle.load(open('small_quality_0_pseudolabel.pkl', "rb" )) 

            self.db = pickle.load(open('quality_RGU_iter0_1.pkl', "rb" )) 
            # self.db = pickle.load(open('small_quality_0.pkl', "rb" )) 
            # self.db = pickle.load(open('noU_3.pkl', "rb" ))
            # self.db = pickle.load(open('pseudo_curated_2_v3.pkl', "rb" ))

        Y = None
        index_array = np.arange(len(data_path))
        self.augment = augment
        self.get_label = get_label

        while(True):
            batches = self.make_batches(len(data_path), batch_size)
            if(training):
               index_array =  shuffle(index_array, random_state=42)

            # if(training or get_label == False):
            #     index_array = shuffle(index_array, random_state=42)

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]

                from multiprocessing.dummy import Pool
                p = Pool(4)
                Xf = p.map(self.load_image, [data_path[x] for x in batch_ids])

                if(self.cfg.quality):
                    X = np.array([i[0] for i in Xf])
                    F = np.array([i[1] for i in Xf])
                else:
                    X = Xf
                    
                if(get_label):
                    Y = np.array([self.query_label(data_path[y], training)
                                for y in batch_ids], dtype = np.float32)
                else: 
                    Y = np.array([data_path[y] for y in batch_ids])

                p.close()
                # if(augment and get_label):
                #     X, Y = self.augment_batch(X, Y)
                X = np.array(X) 
                if(self.cfg.quality):
                    z = []

                    for k, x in enumerate(batch_ids):
                        path_q = data_path[x].split('/')[1].split('.')[0]
                        classes = data_path[x].split('/')[0]
                        z.append(F[k])

                    # for k, x in enumerate(batch_ids):
                    #     path_q = data_path[x].split('/')[1].split('.')[0]
                    #     classes = data_path[x].split('/')[0]
                    #     if(classes != 'G'):
                    #         z.append(self.Z[path_q])
                    #     else:
                    #         z.append(self.mu) #q2
                    z = np.array(z)

                if(cfg.return_index):
                    yield X, Y, [data_path[x] for x in batch_ids]
                else:
                    if(self.cfg.quality):
                        yield X, Y, z
                    else:
                        yield X, Y
                    # print(X.shape)
                    

            if(not training):
                yield None

    def load_image(self, img_path):
        """Returns an image give a file path.
        # Arguments
            img_path: String, path of an image.
            target_size: (Int, Int), the final size of the image (H, W).
        # Returns
            Array(H x W x 3) an read image from the given datapath. The image is in a RGB format.
        """
        from PIL import Image, ImageOps
        fl_path = '/'.join(img_path.split('/')[:5] + ['fluorescence'] + img_path.split('/')[6:])
        image =  np.array(Image.open(img_path), dtype = np.float32)
        fl_image = np.array(Image.open(fl_path), dtype = np.float32)
        image = np.concatenate([image, fl_image], axis = 2)

        if(self.augment and self.get_label):
            image = self.actual_aug(image)

        if(self.cfg.quality):
            score = self.Z[img_path.split('/')[-1].split('.')[0]]
            # score = self.calculate_metric(image[..., :3], self.Z['params'])
            # score = (score + 9.315512) / (9.315512 + 9.01491)
        assert not(self.cfg.read_U == True and self.cfg.read_RGU == True)

        if(self.cfg.read_U):
            image[..., 2] = image[..., 5]
            image = image[..., :3]
        elif(self.cfg.read_RGU):
            pass
        else:
            image = image[..., :3]

        # print(image.shape)
        if(self.resize_mode == 'pad'):
            if(self.cfg.read_RGU):
                image_br = image[..., :3]
                image_fl = image[..., 3:]
                image_br = Image.fromarray(np.uint8(image_br))
                image_br = ImageOps.pad(image_br, (self.target_size[0], self.target_size[1]), color='white')

                image_fl = Image.fromarray(np.uint8(image_fl))
                image_fl = ImageOps.pad(image_fl, (self.target_size[0], self.target_size[1]), color='white')
                image = np.concatenate([image_br, image_fl], axis = 2)
            else:
                image = Image.fromarray(np.uint8(image))
                image = ImageOps.pad(image, (self.target_size[0], self.target_size[1]), color='white')
        elif(self.resize_mode == 'def'):
            image = cv2.resize(imgae, (self.target_size[0], self.target_size[1]))
        # image = np.array(cv2.resize(image, (self.target_size[0], self.target_size[1])), dtype = np.float32)#np.array(cv2.resize(image, (self.target_size[0], self.target_size[1]) ) , dtype=np.float32)
        # kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        # image = cv2.filter2D(image, -1, kernel)
        image = np.array(image, dtype = np.float32) / 255
        # print(image.shape)

        if(self.cfg.quality):
            return image, score
        else:
            return image
