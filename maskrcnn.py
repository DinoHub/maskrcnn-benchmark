import torch
from predictor import COCODemo
from maskrcnn_benchmark.config import cfg

COCO_PERSON_INDEX = 1

class MaskRCNN(object):
    def __init__(self, confidence_threshold=0.7):
        cfg.merge_from_file('e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml')
        cfg.MODEL.DEVICE
        cfg.freeze()
        self.model_wrapper = COCODemo(
                        cfg,
                        confidence_threshold=confidence_threshold,
                                    )

    def get_chips_and_masks(self, img, label_index=COCO_PERSON_INDEX):
        '''
        Params
        ------
        img : nd array like, RGB
        label_index : int, index of label wanted

        Returns
        -------
        list of tuple (chip, mask)
        - chip is a ndarray: bb crop of the image
        - mask is a ndarray: same shape as chip, whose 'pixel' value is either 0 or 1, indicating if that pixel belongs to that class or not. 
        '''

        preds = self.model_wrapper.compute_prediction(img)
        top_preds = self.model_wrapper.select_top_predictions(preds)

        labels = top_preds.get_field('labels')
        person_bool_mask = (labels==label_index).numpy().astype(bool)

        masks = top_preds.get_field('mask').numpy()[person_bool_mask]
        bboxes = top_preds.bbox.to(torch.int64).numpy()[person_bool_mask]

        results = []

        for mask, box in zip( masks, bboxes ):
            thresh = mask[0, :, :, None]
            # l,t,r,b = box.to(torch.int64).numpy()
            l,t,r,b = box
            if b - t <= 0 or r - l <= 0:
                continue

            content = img[ t:(b+1), l:(r+1), : ]
            minimask = thresh[ t:(b+1), l:(r+1), : ]
            results.append( (content, minimask) )

        return results                

if __name__ == '__main__':
    import cv2
    # import numpy as np

    maskrcnn = MaskRCNN()
    img_path = '/home/dh/Pictures/studio8-30Nov18/DSC03887.JPG'
    img = cv2.imread(img_path)
    # masks, bboxes = 
    chipsandmasks = maskrcnn.get_chips_and_masks(img)
    print(len(chipsandmasks))

    for chip, mask in chipsandmasks:
        masked = chip * mask
        cv2.imshow( '', masked )
        cv2.waitKey(0)

    # input()