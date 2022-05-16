from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.IN_SUPPORT_FEATURES = ["p5"]

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True

_CC.MODEL.FPN.SUPPORT_LAYER = "p5"
_CC.MODEL.ATTENTION = CN()
_CC.MODEL.ATTENTION.INNER_SIZE = 64
_CC.MODEL.ATTENTION.SELF_ATTENTION_WEIGHT = 0.1

_CC.INPUT.FS = CN()
_CC.INPUT.FS.SUPPORT_WAY = 1
_CC.INPUT.FS.SUPPORT_SHOT = 1
_CC.INPUT.FS.MAX_SUPPORT_OBJ_SIZE = 320
