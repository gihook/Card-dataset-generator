from draw_convex_hulls import get_convex_hull
from draw_rectangles import upper_left_rectangle, upper_right_rectangle
from draw_rectangles import bottom_left_rectangle, bottom_right_rectangle
from util.image_generating import generated_image_height, generated_image_width
from imgaug import Keypoint, KeypointsOnImage
from imgaug import BoundingBox, BoundingBoxesOnImage


def get_keypoints(convex_hull, center_x, center_y):
    hull_points = convex_hull.reshape(-1, 2)
    key_points = [
        Keypoint(x=p[0] + center_x, y=p[1] + center_y) for p in hull_points
    ]
    key_points = KeypointsOnImage(key_points,
                                  shape=(generated_image_width,
                                         generated_image_height, 3))

    return key_points


def get_boundingbox(keypoints_on_image, padding=3, label=""):
    kpsx = [kp.x for kp in keypoints_on_image.keypoints]
    minx = max(0, int(min(kpsx) - padding))
    maxx = min(generated_image_width, int(max(kpsx) + padding))
    kpsy = [kp.y for kp in keypoints_on_image.keypoints]
    miny = max(0, int(min(kpsy) - padding))
    maxy = min(generated_image_height, int(max(kpsy) + padding))

    if minx == maxx or miny == maxy:
        return None

    return BoundingBox(x1=minx, y1=miny, x2=maxx, y2=maxy, label=label)


def get_bounding_boxes_on_image(card_image, image, label):
    top_left_convex_hull = get_convex_hull(card_image, upper_left_rectangle)
    top_right_convex_hull = get_convex_hull(card_image, upper_right_rectangle)
    bottom_left_convex_hull = get_convex_hull(card_image,
                                              bottom_left_rectangle)
    bottom_right_convex_hull = get_convex_hull(card_image,
                                               bottom_right_rectangle)

    card_height, card_width = card_image.shape[:2]
    image_height, image_width = image.shape[:2]
    print(card_width, card_height, image_height, image_width)
    center_x = int((image_width - card_width) / 2)
    center_y = int((image_height - card_height) / 2)

    hulls = [
        top_left_convex_hull, top_right_convex_hull, bottom_left_convex_hull,
        bottom_right_convex_hull
    ]

    bounding_boxes = [
        get_boundingbox(get_keypoints(hull, center_x, center_y), label=label)
        for hull in hulls
    ]

    bounding_boxes = filter(lambda x: x is not None, bounding_boxes)
    bounding_boxes = list(bounding_boxes)
    bounding_boxes_on_image = BoundingBoxesOnImage(bounding_boxes,
                                                   shape=image.shape)

    return bounding_boxes_on_image
