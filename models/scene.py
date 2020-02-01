import cv2
from util.image_generating import card_with_empty_backgroud
from util.bbs_generating import get_bounding_boxes_on_image


class Scene:
    def __init__(self, image, bbs):
        self.bbs = bbs
        self.image = image

    @classmethod
    def from_path(cls, path: str):
        card_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = card_with_empty_backgroud(card_image)
        label = path.split("/")[-1].replace(".png", "")
        bbs = get_bounding_boxes_on_image(card_image, image, label)

        return cls(image, bbs)
