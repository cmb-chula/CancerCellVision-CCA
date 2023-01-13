import pickle 
from pascal_voc_writer import Writer
import cv2
import os
import shutil
import copy
import argparse

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

base_path = '../detection/mmdetection/data/dataset/other_project/CTC_final/brightfield/'
fl_path = '../detection/mmdetection/data/dataset/other_project/CTC_final/fluorescence/'

eval_dict = {'raw' : {}}
dst = 'data/dataset/CTC_image_for_classification_inference'
if(os.path.exists(dst)):
    shutil.rmtree(dst)


with open('../detection/mmdetection/data/dataset/other_project/CTC_final/test.txt', "rb") as f:
    scale = 1 
    
    for pred_bbox, file in zip(base_pkl, f.readlines()):
        file_name = file.strip().decode()
        brightfield_path = base_path + file_name + '.tiff'
        output_path = base_path + 'output/' + file_name + '.tiff'
        img = cv2.imread(brightfield_path)
        fl_img = cv2.imread(fl_path + file_name + '.tiff')

        for class_id, bbox_list in enumerate(pred_bbox):
            for obj_id, bbox in enumerate(bbox_list):
                xmin, ymin, xmax, ymax, conf = bbox
                xmin, ymin, xmax, ymax = int(xmin * scale), int(ymin * scale), int(xmax * scale), int(ymax * scale)
                w = xmax - xmin
                h = ymax - ymin
                xmin = max(0, int(xmin))
                ymin = max(0, int(ymin))
                xmax = min(1080, int(xmax))
                ymax = min(1080, int(ymax ))

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

                pos_conf = conf if class_id ==0 else 1 - conf
                xmin, ymin, xmax, ymax, conf = bbox

                if(path not in eval_dict['raw']):
                    eval_dict['raw'][path] = []
                eval_dict['raw'][path] = [xmin, ymin, xmax, ymax, pos_conf] 

                file_name = path.split('_')[0]
                if(file_name not in eval_dict):
                    eval_dict[file_name] = {'pred' : [], 'gt' : []}
                eval_dict[file_name]['pred'].append([xmin, ymin, xmax, ymax, pos_conf])

for file_name in eval_dict:
    if(file_name == 'raw'):continue
    xml_path = '../detection/mmdetection/data/dataset/other_project/CTC_final/Annotations/' + file_name + '.xml'
    annotations = extract_label_from_voc(xml_path)
    for bbox in annotations:
        xmin, ymin, xmax, ymax, obj_cls = bbox
        if(obj_cls == target_cls):
            eval_dict[file_name]['gt'].append([xmin, ymin, xmax, ymax])

pickle.dump(eval_dict, open(args["output_path"], "wb"))    # print(annotations)
# print(eval_dict)