import logging
import copy

import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.logger import log_first_n
from detectron2.layers import ShapeSpec
from torch import nn

from fsdet.modeling.roi_heads import build_roi_heads
from fsdet.modeling.attention import CrossAttention
# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        # print(self.backbone.output_shape().items())
        input_shape = {}
        for key, val in self.backbone.output_shape().items():
            input_shape[key] = ShapeSpec(channels=val.channels*2, height=val.height, width=val.width, stride=val.stride)

        # print(input_shape)
        # print(self.backbone.output_shape().items())
        self.proposal_generator = build_proposal_generator(
            cfg, input_shape
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        self.support_way = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot = cfg.INPUT.FS.SUPPORT_SHOT
        self.support_layer = cfg.MODEL.FPN.SUPPORT_LAYER

        self.cross_attention = CrossAttention(cfg, self.backbone.output_shape()[self.support_layer].channels)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def batched_cross_attention(self, batched_features, batched_supports):
        """
        batched_features: (B, C, H, W)
        batched_supports: (B, N, C, H, W)
        """
        all_support_ft = []
        for features, supports in zip(batched_features, batched_supports):
            support_ft = self.cross_attention(features[None,...], supports)
            all_support_ft.append(support_ft)

        batched_support_ft = torch.cat(all_support_ft, axis=0)
        # batched_support_ft shape: (B, C, Hq, Wq)
        return batched_support_ft

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * pos_instances (optional): groundtruth :class:`Instances`
                * neg_instances (optional): similar to pos_instances
                * instances (optional): all instances
                * support_images (optional): support images associated with this query (when training)
                * support_boxes (optional): support boxes associated with this query (when training)
                * support_cls (optional): support classes associated with this query (when training)
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "pos_instances" in batched_inputs[0]:
            pos_gt_instances = [
                x["pos_instances"].to(self.device) for x in batched_inputs
            ]
            pos_cls = batched_inputs[0]['support_pos_cls']
        else:
            pos_gt_instances = None
            pos_cls = None

        if "neg_instances" in batched_inputs[0]:
            neg_gt_instances = [
                x["neg_instances"].to(self.device) for x in batched_inputs
            ]
            neg_cls = batched_inputs[0]['support_neg_cls']
        else:
            neg_gt_instances = None
            neg_cls = None

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # print(features.keys())
        pos_features, neg_features = {}, {}

        pos_supports, pos_image_sizes = self.process_supports(batched_inputs, "support_pos_images")
        neg_supports, neg_image_sizes = self.process_supports(batched_inputs, "support_neg_images")

        if len(pos_supports.shape)!=5: 
            print(pos_supports.shape)
            continue
        B, N, C, H, W = pos_supports.shape
        if N!=self.support_shot: 
            print(pos_supports.shape)
            continue
        if C==0 or H==0 or W==0: 
            print(pos_supports.shape)
            continue

        if len(neg_supports.shape)!=5: 
            print(neg_supports.shape)
            continue
        B, N, C, H, W = neg_supports.shape
        if N!=self.support_shot: 
            print(neg_supports.shape)
            continue
        if C==0 or H==0 or W==0: 
            print(neg_supports.shape)
            continue

        # print(f'pos_image_sizes: {pos_image_sizes[0]}')
        # # shape: (B, N, C, H, W)
        # print("backbone grad")
        # print(self.backbone.bottom_up.res5[2].conv3.weight.grad)
        # print("fpn grad")
        # print(self.backbone.fpn_output5.weight.grad)
        # print("conv grad")
        # print(self.cross_attention.conv_query.weight.grad)
        # print("rpn grad")
        # print(self.proposal_generator.rpn_head.conv.weight.grad)
        # print("conv grad 2")
        # print(self.roi_heads.cross_attention.conv_query.weight.grad)
        # print("roi grad")
        # print(self.roi_heads.box_head.fc1.weight.grad)
        # print(self.roi_heads.box_head.fc2.weight.grad)
        # print("cls grad")
        # print(self.roi_heads.box_predictor.cls_score.weight.grad)
        # print("regress grad")
        # print(self.roi_heads.box_predictor.bbox_pred.weight.grad)
        for key, feature in features.items():
            # shape: (B, C, H, W)
            pos_supports_fts = self.batched_cross_attention(feature, pos_supports)
            neg_supports_fts = self.batched_cross_attention(feature, neg_supports)
            # shape: (B, C, H, W)
            # print(f'pos_supports_fts shape: {pos_supports_fts.shape}')
            pos_features[key] = torch.cat([feature, pos_supports_fts], dim=1)
            neg_features[key] = torch.cat([feature, neg_supports_fts], dim=1)
        
        pos_proposals, pos_proposal_loss = self.rpn_forward(images, pos_features, pos_gt_instances, batched_inputs)
        neg_proposals, neg_proposal_loss = self.rpn_forward(images, neg_features, neg_gt_instances, batched_inputs)  # neg_gt_instances should be emtp empty

        #all_proposals = [Instances.cat([pos_proposal, neg_proposal]) for pos_proposal, neg_proposal in zip(pos_proposals, neg_proposals)]
        with torch.no_grad():
            pos_support_proposals = []  # fake proposals: just using it to crop the entire region to 7 by 7
            neg_support_proposals = []
            for b in range(len(pos_proposals)):
                pos_img_height = pos_image_sizes[b][0]
                pos_img_width = pos_image_sizes[b][1]
                for i in range(self.support_shot):
                    instances = Instances(image_size=(pos_img_height, pos_img_width))
                    instances.proposal_boxes = Boxes(torch.tensor([[0, 0, pos_img_width, pos_img_height]])).to(self.device)
                    pos_support_proposals.append(instances)

                neg_img_height = neg_image_sizes[b][0]
                neg_img_width = neg_image_sizes[b][1]
                for i in range(self.support_shot):
                    instances = Instances(image_size=(neg_img_height, neg_img_width))
                    instances.proposal_boxes = Boxes(torch.tensor([[0, 0, neg_img_width, neg_img_height]])).to(self.device)
                    neg_support_proposals.append(instances)

        proposal_losses = {key : pos_proposal_loss[key]+neg_proposal_loss[key] for key in pos_proposal_loss.keys()}

        _, pos_detector_losses = self.roi_heads(  
            images, features, pos_proposals, pos_supports, pos_support_proposals, targets=pos_gt_instances, pref_cls=pos_cls
        )

        _, neg_detector_losses = self.roi_heads(  
            images, features, neg_proposals, neg_supports, neg_support_proposals, targets=pos_gt_instances, pref_cls=neg_cls
        )

        detector_losses = {key : pos_detector_losses[key]+neg_detector_losses[key] for key in pos_detector_losses.keys()}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def rpn_forward(self, images, features, gt_instances, batched_inputs):
        """
        return:
            proposal: List
            proposal_losses: Dict
        """
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        return proposals, proposal_losses

    def init_model(self, batched_inputs):
        """
        batched_inputs is a list of dict --- outputs of DatasetMapper
        """
        self.supports, self.supports_image_sizes = self.process_supports(batched_inputs, "support_images")
        print(self.supports.shape)
        # self.supports.shape (Cl, N, C, H, W)

    def uninit_model(self):
        del self.supports
        del self.supports_image_sizes

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        all_proposals = None
        all_logits = None
        all_deltas = None
        for ci, cls_support_fts in enumerate(self.supports):
            cat_features = {}
            for key, feature in features.items():
                # shape: (B, C, H, W)
                supports_fts = self.cross_attention(feature, cls_support_fts)
                # shape: (B, C, H, W)
                cat_features[key] = torch.cat([feature, supports_fts], dim=1)
            proposals, _ = self.rpn_forward(images, cat_features, None, None)

            support_img_height = self.supports_image_sizes[ci][0]
            support_img_width = self.supports_image_sizes[ci][1]
            support_proposals = []
            instances = Instances(image_size=(support_img_height, support_img_width))
            instances.proposal_boxes = Boxes(torch.tensor([[0, 0, support_img_width, support_img_height]], device=self.device)).to(self.device)
            for i in range(self.support_shot):
                support_proposals.append(copy.deepcopy(instances))

            logits, deltas = self.roi_heads(images, features, proposals, cls_support_fts[None, ...], support_proposals, targets=None, pref_cls=ci)
            # shape: (B, Bo, *)
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat([all_logits, logits], dim=1)

            if all_deltas is None:
                all_deltas = deltas
            else:
                all_deltas = torch.cat([all_deltas, deltas], dim=1)

            if all_proposals is None:
                all_proposals = proposals
            else:
                all_proposals = [Instances.cat([prop1s, prop2s]) for prop1s, prop2s in zip(all_proposals, proposals)]

        all_logits = all_logits.reshape(-1, all_logits.shape[-1])
        all_deltas = all_deltas.reshape(-1, all_deltas.shape[-1])
        results = self.roi_heads.aggregate_results(all_logits, all_deltas, all_proposals)
        # print("finish head")
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results


    def process_supports(self, batched_inputs, img_key : str):
        support_images = self.preprocess_support_image(batched_inputs, img_key)
        image_sizes = support_images.image_sizes

        B, N, C, H, W = support_images.tensor.shape
        assert N == self.support_shot

        support_images = support_images.tensor.reshape(B*N, C, H, W)
        # it's fine to turn it into a tensor because all the support images already have the max_size aligned
        # so it's likely that the resulting H, W is close to what a square image would give
        support_features = self.backbone(support_images)[self.support_layer]
        Lf, Cf, Hf, Wf = support_features.shape

        assert Lf == B*N
        return support_features.reshape(B, N, Cf, Hf, Wf), image_sizes


    def preprocess_support_image(self, batched_inputs, img_key : str):
        support_images = [[self.normalizer(img.to(self.device)) for img in x[img_key]] for x in batched_inputs]
        support_images = [ImageList.from_tensors(imgs, self.backbone.size_divisibility).tensor for imgs in support_images]
        support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
        return support_images


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
