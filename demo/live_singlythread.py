import argparse
import cv2

from threading import Thread, Lock
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from copy import deepcopy
import time


parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument(
    "--config-file",
    default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.7,
    help="Minimum score for the prediction to be shown",
)
parser.add_argument(
    "--min-image-size",
    type=int,
    default=224,
    help="Smallest size of the image to feed to the model. "
        "Model was trained with 800, which gives best results",
)
parser.add_argument(
    "--show-mask-heatmaps",
    dest="show_mask_heatmaps",
    help="Show a heatmap probability for the top masks-per-dim masks",
    action="store_true",
)
parser.add_argument(
    "--masks-per-dim",
    type=int,
    default=2,
    help="Number of heatmaps per dimension to show",
)
parser.add_argument(
    "opts",
    help="Modify model config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

# load config from file and command-line arguments
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()


coco_demo = COCODemo(
    cfg,
    confidence_threshold=args.confidence_threshold,
    show_mask_heatmaps=args.show_mask_heatmaps,
    masks_per_dim=args.masks_per_dim,
    # min_image_size=args.min_image_size,
)

cam = cv2.VideoCapture(0)
cv2_win = 'MASKRCNN (COCO)'
cv2.namedWindow(cv2_win, cv2.WINDOW_NORMAL)

while True:
    ret, img = cam.read()


    preds = coco_demo.compute_prediction(img)
    top_preds = coco_demo.select_top_predictions(preds)

    masks = top_predictions.get_field('mask')
    labels = top_predictions.get_field('labels')
    bboxes = top_predictions.bbox
    
    person_bool_mask = (labels == COCO_PERSON_INDEX ).numpy().astype(bool)

    img_show = deepcopy(img)

    cv2.imshow(cv2_win, img_show)
    if cv2.waitKey(1) == ord('q'):
        break  # esc to quit


cv2.destroyAllWindows()
