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

    def get_bb_and_mask(self, img):
        preds = self.model_wrapper.compute_prediction(img)
        top_preds = self.model_wrapper.select_top_predictions(preds)

        masks = top_preds.get_field('mask')
        labels = top_preds.get_field('labels')
        bboxes = top_preds.bbox
    
        person_bool_mask = (labels == COCO_PERSON_INDEX ).numpy().astype(bool)

        return

if __name__ == '__main__':
    import cv2

    maskrcnn = MaskRCNN()
    img = cv2.imread('/home/dh/Pictures/bowen.png')
    maskrcnn.get_bb_and_mask(img)
    input()