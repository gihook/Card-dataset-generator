import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from resizing_images import card_height, card_width
from process_image import display_image_and_wait
import numpy as np
from draw_convex_hulls import get_convex_hull
from draw_rectangles import upper_left_rectangle, upper_right_rectangle, bottom_left_rectangle, bottom_right_rectangle
import os

generated_image_height = 720
generated_image_width = 720
center_x = int((generated_image_width - card_width) / 2)
center_y = int((generated_image_height - card_height) / 2)


def get_keypoints(convex_hull, center_x=center_x, center_y=center_y):
    hull_points = convex_hull.reshape(-1, 2)
    key_points = [
        Keypoint(x=p[0] + center_x, y=p[1] + center_y) for p in hull_points
    ]
    key_points = KeypointsOnImage(key_points,
                                  shape=(generated_image_width,
                                         generated_image_height, 3))

    return key_points


def get_boundingbox(keypoints, padding=3, label=None):
    kpsx = [kp.x for kp in keypoints.keypoints]
    minx = max(0, int(min(kpsx) - padding))
    maxx = min(generated_image_width, int(max(kpsx) + padding))
    kpsy = [kp.y for kp in keypoints.keypoints]
    miny = max(0, int(min(kpsy) - padding))
    maxy = min(generated_image_height, int(max(kpsy) + padding))

    if minx == maxx or miny == maxy:
        return None

    return BoundingBox(x1=minx, y1=miny, x2=maxx, y2=maxy, label=label)


def centered_card(card_image):
    card_height, card_width = card_image.shape[:2]

    image_center_x = int(generated_image_width / 2)
    image_center_y = int(generated_image_height / 2)

    card_center_x = int(card_width / 2)
    card_center_y = int(card_height / 2)

    empty_image = np.zeros((generated_image_height, generated_image_width, 4),
                           dtype=np.uint8)
    empty_image[(image_center_y - card_center_y):(image_center_y +
                                                  card_height - card_center_y),
                (image_center_x - card_center_x):(card_width + image_center_x -
                                                  card_center_x)] = card_image

    return empty_image


def get_bounding_boxes_on_image(card_image, image, label):
    top_left_convex_hull = get_convex_hull(card_image, upper_left_rectangle)
    top_right_convex_hull = get_convex_hull(card_image, upper_right_rectangle)
    bottom_left_convex_hull = get_convex_hull(card_image,
                                              bottom_left_rectangle)
    bottom_right_convex_hull = get_convex_hull(card_image,
                                               bottom_right_rectangle)

    bounding_boxes = [
        get_boundingbox(get_keypoints(top_left_convex_hull), label=label),
        get_boundingbox(get_keypoints(top_right_convex_hull), label=label),
        get_boundingbox(get_keypoints(bottom_left_convex_hull), label=label),
        get_boundingbox(get_keypoints(bottom_right_convex_hull), label=label)
    ]

    bounding_boxes = filter(lambda x: x is not None, bounding_boxes)
    bounding_boxes = list(bounding_boxes)
    bounding_boxes_on_image = BoundingBoxesOnImage(bounding_boxes,
                                                   shape=image.shape)

    return bounding_boxes_on_image


def get_paths_and_names(folder_path):
    folder_files = os.listdir(folder_path)
    folder_files = sorted(folder_files)
    pwd = os.getcwd()
    image_paths = map(
        lambda file_name:
        (os.path.join(pwd, folder_path, file_name), file_name), folder_files)

    return image_paths


def main():
    image = cv2.imread("sarme.jpg")
    image = image[:generated_image_width, :generated_image_height]

    folder_path = "resized_images_with_alphachannel/"
    image_paths = get_paths_and_names(folder_path)

    for (image_path, file_name) in image_paths:
        card_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        empty_image = centered_card(card_image)
        bounding_boxes = get_bounding_boxes_on_image(card_image, image,
                                                     file_name)

        seq = iaa.Sequential([
            iaa.Affine(translate_percent={
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2)
            },
                       rotate=(-180, 180),
                       scale=(0.3, 1)),
        ])
        image_aug, bbs_aug = seq(image=empty_image,
                                 bounding_boxes=bounding_boxes)

        _, _, _, alphachannel = cv2.split(image_aug)
        alphachannel = 255 - alphachannel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result = cv2.bitwise_or(image.copy(), image.copy(), mask=alphachannel)
        result = cv2.bitwise_or(result, image_aug)

        result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        result = bbs_aug.draw_on_image(result, size=2)

        display_image_and_wait(result)


if __name__ == '__main__':
    main()
