# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 23:07:05 2022

@author: tripa
"""

import os
import collections
from xml.etree.ElementTree import parse
from PIL import Image
        
class VOCDataloader():
    
    def __init__(self, voc_root, image_set='train', transforms=None):
        
        # valid_image_sets = ["train", "trainval", "val", "test"]
        self.image_set = image_set
        self.transforms = transforms
        
        splits_dir = os.path.join(voc_root, "ImageSets", "Main")
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        
        target_dir = os.path.join(voc_root, "Annotations")
        self.targets = [os.path.join(target_dir, x + ".xml") for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self):
        return len(self.images)
    
    @property
    def annotations(self):
        return self.targets
    
    def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is a dictionary of the XML tree.
            """
            
            img = Image.open(self.images[index]).convert("RGB")
            target = self.parse_voc_xml(parse(self.annotations[index]).getroot())

            if self.transforms is not None:
                # Here, the transformations will apply only to image and not the labels
                img, target = self.transforms(img)

            return img, target
    
    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict