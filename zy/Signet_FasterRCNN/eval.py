from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger

save_dir = "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FasterRCNN/tmp"
logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())

config_file = "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FasterRCNN/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
model = build_detection_model(cfg)
model.eval()
device = torch.device(cfg.MODEL.DEVICE)
save_dir = cfg.OUTPUT_DIR
checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir)
_ = checkpointer.load(cfg.MODEL.WEIGHT)
cpu_device = torch.device("cpu")

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

def build_transform(cfg):
    """
    Creates a basic transformation that was used to train the models
    """
    #cfg = self.cfg

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T.Compose(
        [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

transforms = build_transform(cfg)
#original_image = Image.open('/home/zhongying/reference/segment/maskrcnn-benchmark/datasets/Signet_ring_cell_dataset/sig-train-pos/2018_64982_1-3_2019-02-25 21_57_36-lv0-33516-59515-2003-2010.jpeg').convert("RGB")
#original_image = Image.open('/home/zhongying/reference/segment/maskrcnn-benchmark/datasets/Signet_ring_cell_dataset/sig-train-neg/D20190481111_2019-06-10 09_22_49-lv0-61840-23896-2000-2000.jpeg').convert("RGB")
original_image = Image.open('/home/zhongying/reference/segment/maskrcnn-benchmark/datasets/Signet_ring_cell_dataset/sig-train-pos/2018_67251_1-3_2019-02-26 00_02_22-lv0-29705-20186-2024-2049.jpeg').convert("RGB")
original_image = np.array(original_image)[:, :, [2, 1, 0]]
image = transforms(original_image)
image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(device)

with torch.no_grad():
    predictions = model(image_list)
predictions = [o.to(cpu_device) for o in predictions]
prediction = predictions[0]
height, width = original_image.shape[:-1]
prediction = prediction.resize((width, height))

def select_top_predictions(predictions, confidence_threshold = 0.7):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

top_predictions = select_top_predictions(prediction)

result = original_image.copy()

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    #colors = self.compute_colors_for_labels(labels).tolist()

    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), (0, 0, 0), 5
        )

    return image

result = overlay_boxes(result, top_predictions)

cv2.imwrite('/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FasterRCNN/tmp/result.jpeg', result)
#cv2.waitKey(0)

print('1')
