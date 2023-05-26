'''
One class AP evaluation code. 
The code was modified from https://github.com/Cartucho/mAP/blob/master/main.py
'''
import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math

import numpy as np
import pickle 
from pascal_voc_writer import Writer
from torchvision.ops import nms
import torch

MINOVERLAP = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("-ip", "--input_path", required=True, help="Input pickle file path")
parser.add_argument("-pid", "--patient_id", required=False, help="ID of the patient (for per patient eval)")


args = vars(parser.parse_args())
eval_pickle = pickle.load(open(args["input_path"], "rb"))
eval_pickle.pop('raw')

eval_keys = eval_pickle.keys()
eval_keys = sorted(list(eval_keys))

eval_patient_id = None
if(args["patient_id"] is not None):
    eval_patient_id = args["patient_id"]
    assert eval_patient_id in ['r06', 'r07', 'r08']

# dictionary with counter per class
gt_counter_per_class = {}
counter_images_per_class = {}
gt_boxes = {}
pred_boxes = []

def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


#format pred and GT boxes
for file_name in eval_keys:
    if(eval_patient_id is not None and file_name[:3] != eval_patient_id):
        continue

    gt_bounding_boxes = []
    pred_bounding_boxes = []

    already_seen_classes = []
    class_name = 'R'
    for bbox in eval_pickle[file_name]['gt']:
        xmin, ymin, xmax, ymax= bbox
        gt_bounding_boxes.append({"class_name":class_name, "bbox":[xmin, ymin, xmax, ymax], "used":False})
        if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            gt_counter_per_class[class_name] = 1

        if class_name not in already_seen_classes:
            if class_name in counter_images_per_class:
                counter_images_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                counter_images_per_class[class_name] = 1
            already_seen_classes.append(class_name)
    gt_boxes[file_name] = gt_bounding_boxes
    # print(file_name, len(gt_bounding_boxes), len(eval_pickle[file_name]['gt']))

    pred_bbox = np.array(eval_pickle[file_name]['pred'], dtype = np.float32)
    nms_box_id = nms( torch.tensor(pred_bbox[:, :4]), torch.tensor(pred_bbox[:, 4]), 0.5)
    pred_bbox = pred_bbox[nms_box_id]

    for bbox in pred_bbox:
        xmin, ymin, xmax, ymax, confidence = bbox
        pred_bounding_boxes.append({"confidence":confidence, "file_id":file_name, "bbox":[xmin, ymin, xmax, ymax]})
    # pred_bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
    pred_boxes += pred_bounding_boxes
pred_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)


# print(pred_boxes[0])
# print(gt_boxes[0])
gt_classes = list(gt_counter_per_class.keys())
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
print(gt_counter_per_class)

"""
 Calculate the AP for each class
"""

sum_AP = 0.0
ap_dictionary = {}
lamr_dictionary = {}
# open file to store the output
count_true_positives = {}
for class_index, class_name in enumerate(gt_classes):
    count_true_positives[class_name] = 0
    """
        Assign detection-results to ground-truth objects
    """
    nd = len(pred_boxes)

    tp = [0] * nd # creates an array of zeros of size nd
    fp = [0] * nd

    # print(len(tp), len(fp))
    for idx, detection in enumerate(pred_boxes):
        ground_truth_data = gt_boxes[detection['file_id']]
        ovmax = -1
        gt_match = -1
        # load detected object bounding-box
        bb = detection['bbox']#[i['bbox'] for i in detection]
        # print(ground_truth_data)

        # print(ground_truth_data)
        for obj in ground_truth_data:
            # look for a class_name match
            if obj["class_name"] == class_name:
                bbgt = obj['bbox']

                bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj
        # print(bb, detection['file_id'], ovmax)
        # set minimum overlap
        min_overlap = MINOVERLAP
        if ovmax >= min_overlap:
            # if "difficult" not in gt_match:
            if not bool(gt_match["used"]):
                # true positive
                tp[idx] = 1
                gt_match["used"] = True
                count_true_positives[class_name] += 1
            else:
                fp[idx] = 1
        else:
            # false positive
            fp[idx] = 1
            if ovmax > 0:
                status = "INSUFFICIENT OVERLAP"
    #prin   t(tp)
    # compute precision/recall
    # print(fp)

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    #print(tp)
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
    #print(rec)
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    #print(prec)

    ap, mrec, mprec = voc_ap(rec[:], prec[:])
    sum_AP += ap
    text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)


    """
        Write to output.txt
    """
    import pickle
    hmean = np.max(2 /( (1 / np.array(prec) ) + (1/ np.array(rec)) )) 
    print("F1 : {:.2f}".format(hmean * 100))

    # pickle.dump((prec, rec, tp, fp), open(output_files_path + '/output.pkl', 'wb'))
    rounded_prec = [ '%.2f' % elem for elem in prec ]
    rounded_rec = [ '%.2f' % elem for elem in rec ]

    # output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
    # if not args.quiet:
        # print(text)
    ap_dictionary[class_name] = ap

    n_images = counter_images_per_class[class_name]
    lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
    lamr_dictionary[class_name] = lamr

# output_file.write("\n# mAP of all classes\n")
mAP = sum_AP / n_classes
text = "mAP = {0:.2f}".format(mAP*100)
print(text)