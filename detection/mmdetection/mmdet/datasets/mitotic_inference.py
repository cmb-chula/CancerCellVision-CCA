from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style_legacy import XMLLegacyDataset

import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image


@DATASETS.register_module()
class MitoticInferenceDataset(XMLLegacyDataset):

    CLASSES = ['mitotic']

    def __init__(self, **kwargs):
        super(MitoticInferenceDataset, self).__init__(**kwargs)


    def load_annotations(self, ann_file):
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename =  f'WSI/{img_id}'
            data_infos.append(
                dict(id=img_id, filename=filename, width=512, height=512))
        return data_infos


    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=None,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
    
        return eval_results
