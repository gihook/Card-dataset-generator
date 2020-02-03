import cv2
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
from imgaug import augmenters as iaa
from util.image_generating import card_with_empty_backgroud
from util.bbs_generating import get_bounding_boxes_on_image
from imgaug import BoundingBoxesOnImage


def is_blocked(scene, bbx: BoundingBox):
    x = int(bbx.center_x)
    y = int(bbx.center_y)
    width, height = scene.image.shape[:2]

    if (x > width):
        return False

    if (y > height):
        return False

    pixel = scene.image[y][x]

    return (len(list(filter(lambda x: x != 0, pixel))) > 0)


class Scene:
    def __init__(self, image, bbs):
        self.bbs: BoundingBoxesOnImage = bbs
        self.image = image

    @classmethod
    def from_path(cls, path: str):
        card_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = card_with_empty_backgroud(card_image)
        label = path.split("/")[-1].replace(".png", "")
        bbs = get_bounding_boxes_on_image(card_image, image, label)

        return cls(image, bbs)

    @classmethod
    def from_image(cls, image, bbs):
        return cls(image, bbs)

    def merge(self, overlay):
        _, _, _, alphachannel = cv2.split(overlay.image)
        alphachannel = 255 - alphachannel
        result = cv2.bitwise_or(self.image.copy(),
                                self.image.copy(),
                                mask=alphachannel)
        result = cv2.bitwise_or(result, overlay.image)
        bbs = list(
            filter(lambda bbx: not is_blocked(overlay, bbx),
                   self.bbs.bounding_boxes))
        bbs = bbs + overlay.bbs.bounding_boxes
        bbs = BoundingBoxesOnImage(bbs, self.image.shape)

        return Scene.from_image(result, bbs)

    def random_transformation(self):
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.3, 0.45),
                       translate_percent={
                           "x": (-0.3, 0.3),
                           "y": (-0.3, 0.3)
                       },
                       rotate=(-180, 180))
        ])
        new_image, new_bbs = seq(image=self.image, bounding_boxes=self.bbs)

        return Scene.from_image(new_image, new_bbs)
