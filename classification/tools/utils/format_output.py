import pickle 
from pascal_voc_writer import Writer
import cv2
import os
import shutil
import copy
import argparse
import numpy as np
from torchvision.ops import nms
import torch

target_cls = 'R'

def gen_path(path):
    try: os.makedirs(path)
    except: pass

def extract_label_from_voc(xml_path):
    import xml.etree.ElementTree as ET
    import numpy as np
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for elem in root.iter('object'):
        bbox = [None] * 5
        bbox[4]  = elem[0].text
        for bbox_loc in elem.iter('bndbox'):
            loc = [ int(float(bbox_loc[i].text)) for i in range(4)]
            bbox[:4]= loc
        annotations.append(bbox)
    return annotations


parser = argparse.ArgumentParser()
parser.add_argument("-ip", "--input_path", required=True, help="Input pickle file path")
parser.add_argument("-op", "--output_path", required=False, help="Output pickle file path")
parser.add_argument("-g", "--generate_image", action='store_true', help="Generate image for classification stage")


args = vars(parser.parse_args())
base_pkl = pickle.load(open(args["input_path"], "rb"))
gen_image = args["generate_image"]

base_path = '../detection2/mmdetection/data/dataset/CTC/brightfield/'
fl_path = '../detection2/mmdetection/data/dataset/CTC/fluorescence/'

eval_dict = {'raw' : {}}
dst = 'data/dataset/CTC_image_for_classification_inference'
if(os.path.exists(dst)):
    shutil.rmtree(dst)

with open('../detection2/mmdetection/data/dataset/CTC/test.txt', "rb") as f:
    scale = 1 
    
    for pred_bbox, file in zip(base_pkl, f.readlines()):
        file_name = file.strip().decode()
        brightfield_path = base_path + file_name + '.tiff'
        output_path = base_path + 'output/' + file_name + '.tiff'
        img = cv2.imread(brightfield_path)
        fl_img = cv2.imread(fl_path + file_name + '.tiff')
        scores = pred_bbox['pred_instances']['scores'].reshape(-1, 1)
        bboxes = pred_bbox['pred_instances']['bboxes']
        labels = pred_bbox['pred_instances']['labels'].reshape(-1, 1)
        bboxes = np.concatenate([bboxes, scores, labels], axis = 1)
        # print(file_name, pred_bbox['img_path'], bboxes.shape)
        nms_bboxes = nms(torch.tensor(bboxes[:, :4]), torch.tensor(bboxes[:, 4]), iou_threshold = 0.5)
        bboxes = bboxes[nms_bboxes]

        for obj_id, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax, conf, class_id = bbox
            h = ymax - ymin
            xmin, ymin, xmax, ymax = int(xmin * scale), int(ymin * scale), int(xmax * scale), int(ymax * scale)
            w = xmax - xmin
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(1080, int(xmax))
            ymax = min(1080, int(ymax))
            # print(xmin, ymin, xmax, ymax, conf, file)
            path = '{}_{}'.format(file_name, str(obj_id))
            area = (xmax - xmin) * (ymax - ymin)
            if(not area > 0):
                continue
            
            if(gen_image):
                cell = img[ymin : ymax, xmin : xmax, :]
                fl_cell = fl_img[ymin : ymax, xmin : xmax, :]
                gen_path('{}/brightfield/{}'.format(dst, target_cls))
                gen_path('{}/fluorescence/{}'.format(dst, target_cls))
                cv2.imwrite('{}/brightfield/{}/{}_{}.tiff'.format(dst, target_cls, file_name, str(obj_id)), cell)
                cv2.imwrite('{}/fluorescence/{}/{}_{}.tiff'.format(dst, target_cls, file_name, str(obj_id)), fl_cell)

            pos_conf = conf if class_id == 0 else 1 - conf

            if(path not in eval_dict['raw']):
                eval_dict['raw'][path] = []
            eval_dict['raw'][path] = [xmin, ymin, xmax, ymax, pos_conf] 

            file_name = path.split('_')[0]
            if(file_name not in eval_dict):
                eval_dict[file_name] = {'pred' : [], 'gt' : []}
            eval_dict[file_name]['pred'].append([xmin, ymin, xmax, ymax, pos_conf])
            

for file_name in eval_dict:
    if(file_name == 'raw'):continue
    xml_path = '../detection2/mmdetection/data/dataset/CTC/Annotations/' + file_name + '.xml'
    annotations = extract_label_from_voc(xml_path)
    for bbox in annotations:
        xmin, ymin, xmax, ymax, obj_cls = bbox
        if(obj_cls == target_cls):
            eval_dict[file_name]['gt'].append([xmin, ymin, xmax, ymax])

pickle.dump(eval_dict, open(args["output_path"], "wb"))    # print(annotations)
# print(eval_dict)