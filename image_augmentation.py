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


def get_boundingbox(keypoints, padding=3):
    kpsx = [kp.x for kp in keypoints.keypoints]
    minx = max(0, int(min(kpsx) - padding))
    maxx = min(generated_image_width, int(max(kpsx) + padding))
    kpsy = [kp.y for kp in keypoints.keypoints]
    miny = max(0, int(min(kpsy) - padding))
    maxy = min(generated_image_height, int(max(kpsy) + padding))

    if minx == maxx or miny == maxy:
        return None

    return BoundingBox(x1=minx, y1=miny, x2=maxx, y2=maxy)


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


def main():
    image = cv2.imread("sarme.jpg")
    image = image[:generated_image_width, :generated_image_height]
    empty_image = np.zeros((generated_image_height, generated_image_width, 4),
                           dtype=np.uint8)
    sample_card_path = "resized_images_with_alphachannel/2h.png"
    card_image = cv2.imread(sample_card_path, cv2.IMREAD_UNCHANGED)
    empty_image = centered_card(card_image)

    top_left_convex_hull = get_convex_hull(card_image, upper_left_rectangle)
    top_right_convex_hull = get_convex_hull(card_image, upper_right_rectangle)
    bottom_left_convex_hull = get_convex_hull(card_image,
                                              bottom_left_rectangle)
    bottom_right_convex_hull = get_convex_hull(card_image,
                                               bottom_right_rectangle)

    bounding_boxes = BoundingBoxesOnImage([
        get_boundingbox(get_keypoints(top_left_convex_hull)),
        get_boundingbox(get_keypoints(top_right_convex_hull)),
        get_boundingbox(get_keypoints(bottom_left_convex_hull)),
        get_boundingbox(get_keypoints(bottom_right_convex_hull))
    ],
                                          shape=image.shape)

    seq = iaa.Sequential([iaa.Affine(rotate=(-70, 25), scale=(0.5, 1))])
    image_aug, bbs_aug = seq(image=empty_image, bounding_boxes=bounding_boxes)

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
