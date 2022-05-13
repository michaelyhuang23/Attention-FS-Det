from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True

_CC.MODEL.FPN.SUPPORT_LAYER = "res5"

_CC.INPUT.FS = CN()
_CC.INPUT.FS.SUPPORT_WAY = 1
_CC.INPUT.FS.SUPPORT_SHOT = 1
