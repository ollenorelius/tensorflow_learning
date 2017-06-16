"""Contains all global parameters used in the SqueezeDet project."""
IMAGE_SIZE = 512
IMAGE_CHANNELS = 3

BATCH_SIZE = 32

OUT_CLASSES = 2  # Classes to recognize. 9 for KITTI
OUT_COORDS = 4  # Coordinates, for a box in 2D we need 4.
OUT_CONF = 1  # Number of confidence scores per box.
# Will always be one, but written out for transparecy.

GRID_SIZE = 16  # Number of grid points in which to put anchors in the image.
# MUST MATCH FINAL ACTIVATION SIZE!
# This is in one dimension, so final size is [batch,GRID_SIZE, GRID_SIZE,depth]

ANCHOR_COUNT = 3  # Number of anchors. Must match below.

ANCHOR_SIZES = [[0.8, 0.8],
                [0.4, 0.4],
                [0.2, 0.2]]  # Anchor sizes. [w,h] in relative units

LAMBDA_CONF_P = 75  # Param for weight of used confidence scores
LAMBDA_CONF_N = 100  # Param for punishing unused confidence scores.
LAMBDA_BBOX = 5  # Param weighting in bounding box regression (delta_loss)
