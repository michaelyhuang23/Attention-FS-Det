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
        self.attn_size = cfg.MODEL.ATTENTION.INNER_SIZE
        self.self_attn_weight = cfg.MODEL.ATTENTION.SELF_ATTENTION_WEIGHT

        self.conv_support = nn.Conv2d(self.backbone.output_shape()[self.support_layer].channels, self.attn_size, kernel_size=1, bias=False)
        self.conv_query = nn.Conv2d(self.backbone.output_shape()[self.support_layer].channels, self.attn_size, kernel_size=1, bias=False)
        self.conv_self_support_attn = nn.Conv2d(self.backbone.output_shape()[self.support_layer].channels, 1, kernel_size=1, bias=False)
        # maybe one is enough?

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

    def cross_attention(self, batched_features, batched_supports):
        batched_features_tr = self.conv_query(batched_features)
        all_support_ft = []

        for features_tr, supports in zip(batched_features_tr, batched_supports):
            supports_s = torch.squeeze(self.conv_self_support_attn(supports))
            # self_supports shape: (N, H, W)
            supports_s = supports_s.permute(1,2,0)[None,None,...]

            # supports shape: (N, C, H, W)
            supports_k = self.conv_support(supports)
            supports_k = supports_k.permute(1,2,3,0)
            # supports_k shape: (C', H, W, N)

            # features_tr shape: (C', H, W)
            features_tr = features_tr.permute(1,2,0)
            # features_tr shape: (H, W, C')
            CAtt = torch.tensordot(features_tr-torch.mean(features_tr), supports_k-torch.mean(supports_k), dims=1)
            # CAtt shape: (Hq, Wq, Hs, Ws, Ns)
            CAtt += supports_s * self.self_attn_weight
            CAtt = torch.sigmoid(CAtt)

            CAtt = torch.reshape(CAtt, (CAtt.shape[0], CAtt.shape[1], -1))
            # CAtt shape: (Hq, Wq, L)
            supports_n = supports.permute(2,3,0,1)
            # supports_n shape: (H, W, N, C)
            supports_n = torch.reshape(supports_n, (-1, supports_n.shape[-1]))
            # supports_n shape: (L, C)

            support_ft = torch.tensordot(CAtt, supports_n, dims=1)
            # support_ft shape: (Hq, Wq, C)
            support_ft = support_ft.permute(2, 0, 1)
            all_support_ft.append(support_ft)

        batched_support_ft = torch.stack(all_support_ft, axis=0)
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

        # print("training")
        # print(f'batched_input length: {len(batched_inputs)}')
        # print(batched_inputs[0]['image'].shape, batched_inputs[0]['support_pos_images'][0].shape)

        images = self.preprocess_image(batched_inputs)

        if "pos_instances" in batched_inputs[0]:
            pos_gt_instances = [
                x["pos_instances"].to(self.device) for x in batched_inputs
            ]
        else:
            pos_gt_instances = None

        if "neg_instances" in batched_inputs[0]:
            neg_gt_instances = [
                x["neg_instances"].to(self.device) for x in batched_inputs
            ]
        else:
            neg_gt_instances = None

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
        # print(pos_image_sizes)
        # shape: (B, N, C, H, W)

        for key, feature in features.items():
            # shape: (B, C, H, W)
            pos_supports_fts = self.cross_attention(feature, pos_supports)
            neg_supports_fts = self.cross_attention(feature, neg_supports)
            # shape: (B, C, H, W)
            # print(f'pos_supports_fts shape: {pos_supports_fts.shape}')
            pos_features[key] = torch.cat([feature, pos_supports_fts], dim=1)
            neg_features[key] = torch.cat([feature, neg_supports_fts], dim=1)
        
        pos_proposals, pos_proposal_loss = self.rpn_forward(images, pos_features, pos_gt_instances, batched_inputs)
        neg_proposals, neg_proposal_loss = self.rpn_forward(images, neg_features, neg_gt_instances, batched_inputs)  # neg_gt_instances should be emtp empty

        pos_support_proposals = []  # fake proposals: just using it to crop the entire region to 7 by 7
        for b in range(len(pos_proposals)):
            pos_img_height = pos_image_sizes[b][0]
            pos_img_width = pos_image_sizes[b][1]
            for i in range(self.support_shot):
                instances = Instances(image_size=(pos_img_height, pos_img_width))
                instances.proposal_boxes = Boxes(torch.tensor([[0, 0, pos_img_width, pos_img_height]])).to(self.device)
                pos_support_proposals.append(instances)

        proposal_losses = {key : pos_proposal_loss[key]+neg_proposal_loss[key] for key in pos_proposal_loss.keys()}

        # roi_heads should function as it should in inference: given all gt_instances, it needs to distinguish which class each proposal belongs
        _, detector_losses = self.roi_heads(  
            images, features, pos_proposals, pos_supports, pos_support_proposals, targets=gt_instances
        )

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
        pos_supports, pos_image_sizes = self.process_supports(batched_inputs, "support_pos_images")

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


        
        proposals, _  = self.rpn_forward(images, features, None, batched_inputs)

        results, _ = self.roi_heads(images, features, proposals, None)

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
