import cv2
import time

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = cv2.imread('demo_e2e_mask_rcnn_X_101_32x8d_FPN_1x.png')

start =time.time()
for i in range(100):
    print(i)
    coco_demo.run_on_opencv_image(image)

print('time taken: {}'.format(time.time() - start))