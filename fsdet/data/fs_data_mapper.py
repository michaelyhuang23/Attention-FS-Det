"""
This file contains the datamapper to be passed into the training dataloader
"""

import copy
import logging
import random
from typing import Tuple
import numpy as np
import torch
import cv2
from fvcore.common.file_io import PathManager
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import get_detection_dataset_dicts, DatasetMapper
from detectron2.structures import BoxMode

import pandas as pd
from detectron2.data.catalog import MetadataCatalog

def is_large(anno):
    box = anno['bbox']
    if anno['bbox_mode'] == BoxMode.XYXY_ABS:
        W = max(1, box[2] - box[0])
        H = max(1, box[3] - box[1])
    elif box_mode['bbox_mode'] == BoxMode.XYWH_ABS:
        W = max(1, box[2])
        H = max(1, box[3])
    return W*H>1024

def crop_resize_obj(image, box_in, box_mode, max_size):
    box = copy.deepcopy(box_in)
    box = [round(b) for b in box]
    if box_mode == BoxMode.XYXY_ABS:
        W = max(1, box[2] - box[0])
        H = max(1, box[3] - box[1])
        x1, y1, x2, y2 = box
    elif box_mode == BoxMode.XYWH_ABS:
        W = max(1, box[2])
        H = max(1, box[3])
        x1, y1, x2, y2 = box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1
    else: 
        raise ValueError("wrong box mode")

    x2 = min(image.shape[1]-1, x2)
    y2 = min(image.shape[0]-1, y2)

    assert(x2 >= x1 and y2 >= y1)

    ratio = max_size / float(max(W,H))
    image = image[y1 : y2+1, x1 : x2+1]
    # W and H
    target_size = (round(W*ratio), round(H*ratio))

    box_in[0]=0
    box_in[1]=0
    box_in[2]=target_size[0]
    box_in[3]=target_size[1]

    assert(target_size[0] == max_size or target_size[1] == max_size)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR), box_in

class DatasetFSProcessor:
    def __init__(self, cfg, dataset_name):
        # fmt: off
        self.name = dataset_name
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = False
        self.keypoint_on    = False

        self.support_way    = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot   = cfg.INPUT.FS.SUPPORT_SHOT
        self.max_obj_size   = cfg.INPUT.FS.MAX_SUPPORT_OBJ_SIZE

        if self.max_obj_size == 0 : self.max_obj_size = 320

        assert(self.support_way == 1 or self.support_way == 2)

        dataset_dicts = get_detection_dataset_dicts(
            self.name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        meta_name = self.name[0] if isinstance(self.name, Tuple)  else self.name
        self.classnames = MetadataCatalog.get(meta_name).thing_classes

        self.support_dataset = [{'support_images':[]} for i in range(len(self.classnames))]
        # List[Dict()] a list with each dict corresponding to a class. Inside each dict, we have the 'support_images' key
        for img_dict in dataset_dicts:
            image_orig = utils.read_image(img_dict['file_name'], format=self.img_format)
            anno = img_dict['annotations'][0]
            image, anno['bbox'] = crop_resize_obj(image_orig, anno['bbox'], anno['bbox_mode'], self.max_obj_size)
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            self.support_dataset[anno['category_id']]['support_images'].append(image)

    def get_processed_dataset(self):
        return self.support_dataset


class DatasetMapperWithSupport(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, dataset_name, is_train=True):
        super().__init__(cfg, is_train=is_train)
        # fmt: off
        self.name = dataset_name
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = False
        self.keypoint_on    = False

        self.support_way    = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot   = cfg.INPUT.FS.SUPPORT_SHOT
        self.max_obj_size   = cfg.INPUT.FS.MAX_SUPPORT_OBJ_SIZE

        if self.max_obj_size == 0 : self.max_obj_size = 320

        assert(self.support_way == 1 or self.support_way == 2)

        self.is_train = is_train

        dataset_dicts = get_detection_dataset_dicts(
            self.name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        self.support_dataset = dataset_dicts
        meta_name = self.name[0] if isinstance(self.name, Tuple)  else self.name
        self.classnames = MetadataCatalog.get(meta_name).thing_classes


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # print("loading data")
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # remove useless stuffs
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image


        pos_cls, neg_cls = -1, -1
        if self.is_train:
            # create positive support
            support_pos_images, support_pos_bboxes, support_pos_cls = self.generate_positive_support(dataset_dict)
            pos_cls = support_pos_cls[0]
            dataset_dict['support_pos_images'] = support_pos_images
            dataset_dict['support_pos_cls'] = pos_cls

            if self.support_way == 2:
                # create negative support
                support_neg_images, support_neg_bboxes, support_neg_cls = self.generate_negative_support(dataset_dict)
                neg_cls = support_neg_cls[0]
                dataset_dict['support_neg_images'] = support_neg_images
                dataset_dict['support_neg_cls'] = neg_cls


        image_shape = image.shape[:2]  # h, w

        # one tensor to make it more efficient
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # early return for eval, not needed in our case
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            
            annotation_pos = copy.deepcopy(dataset_dict["annotations"]) # this is consumed
            # USER: Implement additional transformations if you have other types of data
            pos_annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in annotation_pos
                if obj.get("iscrowd", 0) == 0 and obj.get("category_id") == pos_cls
            ]
            pos_instances = utils.annotations_to_instances(
                pos_annos, image_shape
            )
            # Create a tight bounding box from masks, useful when image is cropped
            dataset_dict["pos_instances"] = utils.filter_empty_instances(pos_instances)

            annotation_neg = copy.deepcopy(dataset_dict["annotations"])
            neg_annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in annotation_neg
                if obj.get("iscrowd", 0) == 0 and obj.get("category_id") == neg_cls
            ]
            neg_instances = utils.annotations_to_instances(
                neg_annos, image_shape
            )
            # Create a tight bounding box from masks, useful when image is cropped
            dataset_dict["neg_instances"] = utils.filter_empty_instances(neg_instances)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            all_instances = utils.annotations_to_instances(
                annos, image_shape
            )
            # Create a tight bounding box from masks, useful when image is cropped
            dataset_dict["instances"] = utils.filter_empty_instances(all_instances)

        return dataset_dict

    def generate_positive_support(self, dataset_dict):
        used_categories = list(set([anno['category_id'] for anno in dataset_dict['annotations']]))
        chosen_cls = random.sample(used_categories, 1)[0]
        forbidden_imgs = [dataset_dict['image_id']]
        # print(f"positive chosen id: {chosen_cls}")
        return self.generate_support(chosen_cls, self.support_shot, forbidden_imgs)

    def generate_negative_support(self, dataset_dict):
        used_categories = list(set([anno['category_id'] for anno in dataset_dict['annotations']]))
        not_used_categories = [cls_id for cls_id in range(len(self.classnames)) if cls_id not in used_categories]
        chosen_cls = random.sample(not_used_categories, 1)[0]
        forbidden_imgs = [dataset_dict['image_id']]
        # print(f"negative chosen id: {chosen_cls}")
        return self.generate_support(chosen_cls, self.support_shot, forbidden_imgs)

    def generate_support(self, chosen_cls, shots, forbidden_imgs):
        """
        generate list of support images, support boxes, and support classes.
        the length of the list is = shots
        images are not guaranteed the same shape but is in (C, H, W) format
        """
        # loop through self.support_dataset to find instances (remember the first order is images)
        support_images, support_cls, support_bboxes = [], [], []
        for img_dict in random.sample(self.support_dataset,len(self.support_dataset)):
            if img_dict['image_id'] in forbidden_imgs : continue
            class_annos = [anno for anno in img_dict['annotations'] if anno['category_id'] == chosen_cls]
            class_annos = [anno for anno in class_annos if is_large(anno)]
            if len(class_annos) == 0 : continue
            forbidden_imgs.append(img_dict['image_id'])
            image_orig = utils.read_image(img_dict['file_name'], format=self.img_format)
            if shots - len(support_images) < len(class_annos):
                class_annos = random.sample(class_annos, shots - len(support_images))
            for anno in class_annos:
                image, bboxes = crop_resize_obj(image_orig, anno['bbox'], anno['bbox_mode'], self.max_obj_size)
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                support_images.append(image)
                support_bboxes.append(bboxes)
                assert anno['category_id'] == chosen_cls
                support_cls.append(anno['category_id'])
            if len(support_images) == shots:
                break
        assert len(support_images) == shots
        
        return support_images, support_bboxes, support_cls

