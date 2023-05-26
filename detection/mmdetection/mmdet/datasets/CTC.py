from mmdet.registry import DATASETS
from .xml_style import XMLDataset

import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image
from typing import List, Optional, Union
from mmengine.fileio import get, get_local_path, list_from_file


@DATASETS.register_module()
class CTCDataset(XMLDataset):

    METAINFO = {
            'classes':
            ('R'),
            # ('R', 'G', 'U'),
        }

    # CLASSES = ('R', 'G', 'U')
    def __init__(self,
                 img_subdir: str = 'JPEGImages',
                 ann_subdir: str = 'Annotations',
                 **kwargs) -> None:
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        # print(self.ann_file)
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = f'brightfield/{img_id}.tiff'
            fluorescence_filename = f'fluorescence/{img_id}.tiff'

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] =  osp.join(self.sub_data_root, file_name)
            raw_img_info['fluorescence_file_name'] = osp.join(self.sub_data_root, fluorescence_filename)
            raw_img_info['xml_path'] = osp.join(self.sub_data_root, self.ann_subdir, f'{img_id}.xml')

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = img_info['file_name']
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']
        data_info['fluorescence_img_path'] = img_info['fluorescence_file_name']

        # deal with xml file
        with get_local_path(
                img_info['xml_path'],
                backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            height, width = img.shape[:2]
            del img, img_bytes

        data_info['height'] = height
        data_info['width'] = width

        data_info['instances'] = self._parse_instance_info(
            raw_ann_info, minus_one=True)

        return data_info
