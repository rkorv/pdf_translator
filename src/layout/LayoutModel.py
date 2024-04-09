import os

import cv2
from ultralytics import YOLO


class LayoutModel:
    ENTITIES_COLORS = {
        "caption": (191, 100, 21),
        "footnote": (2, 62, 115),
        "formula": (140, 80, 58),
        "list_item": (168, 181, 69),
        "page_footer": (2, 69, 84),
        "page_header": (83, 115, 106),
        "picture": (255, 72, 88),
        "section-header": (0, 204, 192),
        "table": (116, 127, 127),
        "text": (0, 153, 221),
        "title": (196, 51, 2),
    }
    LABELS_LIST = list(ENTITIES_COLORS)
    BOX_PADDING = 2

    def __init__(
        self,
        models_root="models",
        conf=0.2,
        iou=0.8,
    ):
        self.model = YOLO(os.path.join(models_root, "dlamodel.pt"))

        self.conf = conf
        self.iou = iou

    def draw(self, img, bboxes=[], labels=[], confs=[], only_labels=None):
        for bbox, label, conf in zip(bboxes, labels, confs):
            if only_labels and label not in only_labels:
                continue
            line_thickness = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
            img = cv2.rectangle(
                img=img,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[2], bbox[3]),
                color=self.ENTITIES_COLORS[label],
                thickness=line_thickness,
            )

            text = label + " " + str(conf)
            font_thickness = max(line_thickness - 1, 1)
            (text_w, text_h), _ = cv2.getTextSize(
                text=text, fontFace=2, fontScale=line_thickness / 3, thickness=font_thickness
            )
            img = cv2.rectangle(
                img=img,
                pt1=(bbox[0], bbox[1] - text_h - self.BOX_PADDING * 2),
                pt2=(bbox[0] + text_w + self.BOX_PADDING * 2, bbox[1]),
                color=self.ENTITIES_COLORS[label],
                thickness=-1,
            )
            start_text = (bbox[0] + self.BOX_PADDING, bbox[1] - self.BOX_PADDING)
            img = cv2.putText(
                img=img,
                text=text,
                org=start_text,
                fontFace=0,
                color=(255, 255, 255),
                fontScale=line_thickness / 3,
                thickness=font_thickness,
            )

        return img

    def __call__(self, img):
        inf_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=inf_img, conf=0.2, iou=0.8, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return [], [], []

        res_bboxes, res_labels, res_confs = [], [], []
        for box in boxes:
            detection_class_conf = round(box.conf.item(), 2)
            label = self.LABELS_LIST[int(box.cls)]
            bbox = [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]

            res_bboxes.append(bbox)
            res_labels.append(label)
            res_confs.append(detection_class_conf)

        return res_bboxes, res_labels, res_confs
