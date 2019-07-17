# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from threading import Thread, Lock
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time

mutex = Lock()
mask_thread_dict = {'top_predictions': None, 'frame': None, 'keep_going':True}

def mask_in_thread(coco_demo):

    # print('Initialisting deep sort tracker and its embedder..')
    # tracker = Tracker(max_age = 30, nn_budget = 70) # assuming 7fps & 70nn_budget, tracker looks into 10secs in the past.  
    # print('DeepSORT Tracker inited!')
    # mask_thread_dict['started'] = True
    
    while mask_thread_dict['keep_going']:
        if mask_thread_dict['frame'] is not None:
            mutex.acquire()
            frame = mask_thread_dict['frame'].copy()
            mutex.release()

            predictions = coco_demo.compute_prediction(mask_thread_dict['frame'])
            top_predictions = coco_demo.select_top_predictions(predictions)

            mutex.acquire()
            mask_thread_dict['top_predictions'] = top_predictions
            mutex.release()


def main():
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

    # prepare object that handles inference plus adds predictions on top of image
    # print(args.min_image_size)

    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        # min_image_size=args.min_image_size,
    )

    mask_thread = Thread(target = mask_in_thread, args = (coco_demo,))
    mask_thread.start()

    cam = cv2.VideoCapture(0)
    cv2_win = 'MASKRCNN (COCO)'
    cv2.namedWindow(cv2_win, cv2.WINDOW_NORMAL)

    time_cma = None
    n = 0
    while True:
        ret_val, img = cam.read()

        mutex.acquire()
        mask_thread_dict['frame'] = img
        mutex.release()

        # start_time = time.time()
        # composite = coco_demo.run_on_opencv_image(img)        
        # elapsed = time.time() - start_time
        # if time_cma:
        #     time_cma = time_cma + (elapsed - time_cma) / (n+1)
        # else:
        #     time_cma = elapsed

        composite = img.copy()

        mutex.acquire()
        top_predictions = mask_thread_dict['top_predictions']
        mutex.release()

        if top_predictions:
            if coco_demo.cfg.MODEL.MASK_ON:
                composite = coco_demo.overlay_mask( composite, top_predictions )
            elif coco_demo.show_mask_heatmaps:
                composite = coco_demo.create_mask_montage(composite, top_predictions)

        # print("Time: {:.2f} s / img".format(time.time() - start_time))
        cv2.imshow(cv2_win, composite)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        n += 1
    cv2.destroyAllWindows()
    mask_thread_dict['keep_going'] = False
    print("Time (cma): {:.2f} s / img = {:.2f} FPS".format(time_cma, 1./time_cma))

if __name__ == "__main__":
    main()
