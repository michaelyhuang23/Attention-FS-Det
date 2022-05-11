"""
This file contains the datamapper to be passed into the training dataloader
"""

import copy
import logging
import random
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import get_detection_dataset_dicts

import pandas as pd
from detectron2.data.catalog import MetadataCatalog


class DatasetMapperWithSupport:
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

        # fmt: off
        self.name = dataset_name
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = False
        self.keypoint_on    = False

        self.few_shot       = cfg.INPUT.FS.FEW_SHOT
        self.support_way    = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot   = cfg.INPUT.FS.SUPPORT_SHOT
        self.support_file   = cfg.INPUT.FS.SUPPORT_FILE

        assert(self.support_way == 1 or self.support_way == 2)

        self.is_train = is_train

        dataset_dicts = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        self.support_dataset = dataset_dicts
        self.classnames = MetadataCatalog.get(self.name).thing_classes


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
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

        if self.is_train:
            # create positive support
            support_pos_images, support_pos_bboxes, support_pos_cls = self.generate_positive_support(dataset_dict)
            dataset_dict['support_pos_images'] = support_pos_images
            dataset_dict['support_pos_bboxes'] = support_pos_bboxes
            dataset_dict['support_pos_cls'] = support_pos_cls

            if self.support_way == 2:
                # create negative support
                support_neg_images, support_neg_bboxes, support_neg_cls = self.generate_negative_support(dataset_dict)
                dataset_dict['support_neg_images'] = support_neg_images
                dataset_dict['support_neg_bboxes'] = support_neg_bboxes
                dataset_dict['support_neg_cls'] = support_neg_cls


        image_shape = image.shape[:2]  # h, w

        # one tensor to make it more efficient
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # early return for eval, not needed in our case
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape
            )
            # Create a tight bounding box from masks, useful when image is cropped
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def generate_positive_support(self, dataset_dict):
        used_categories = list(set([anno['category_id'] for anno in dataset_dict['annotations']]))
        chosen_cls = random.sample(used_categories, 1)
        forbidden_imgs = [dataset_dict['image_id']]
        return self.generate_support(chosen_cls, self.support_shot, forbidden_imgs)

    def generate_negative_support(self, dataset_dict):
        used_categories = list(set([anno['category_id'] for anno in dataset_dict['annotations']]))
        not_used_categories = [clsname for clsname in self.classnames if clsname not in used_categories]
        chosen_cls = random.sample(not_used_categories, 1)
        forbidden_imgs = [dataset_dict['image_id']]
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
            if len(class_annos) == 0 : continue
            forbidden_imgs.append(img_dict['image_id'])
            image = utils.read_image(img_dict['file_name'], format=self.img_format)
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if shots - len(support_images) < len(class_annos):
                class_annos = random.sample(class_annos, shots - len(support_images))
            for anno in class_annos:
                support_images.append(image)
                support_bboxes.append(anno['bbox'])
                support_cls.append(anno['category_id'])
        return support_images, support_bboxes, support_cls

