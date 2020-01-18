import cv2
from process_image import display_image_and_wait, merge_images
from draw_rectangles import upper_left_rectangle, upper_right_rectangle, bottom_left_rectangle, bottom_right_rectangle
import numpy as np
import os


def solidity(contour):
    contour_area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return 0

    ratio = contour_area / hull_area

    return ratio


def get_convex_hull(card_image, point):
    top_left_point, bottom_right_point = point

    rectangle_width = bottom_right_point[0] - top_left_point[0]
    rectangle_height = bottom_right_point[1] - top_left_point[1]

    section_x_start = top_left_point[0]
    section_x_end = section_x_start + rectangle_width
    section_y_start = top_left_point[1]
    section_y_end = section_y_start + rectangle_height
    image_section = card_image[section_y_start:section_y_end,
                               section_x_start:section_x_end]

    section_gray = cv2.cvtColor(image_section, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(section_gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    solidities = map(solidity, contours)
    contours = filter(lambda c: solidity(c) > 0, contours)

    single_contour = np.concatenate(list(contours))
    single_hull = cv2.convexHull(single_contour)

    return single_hull + top_left_point


def main():
    folder = "./resized_images_with_alphachannel/"
    card_filenames = os.listdir(folder)
    card_filenames = sorted(card_filenames)
    card_filenames = map(lambda f: os.path.join(folder, f), card_filenames)

    for card_image_path in card_filenames:
        card_image = cv2.imread(card_image_path, cv2.IMREAD_UNCHANGED)
        image_ul = get_convex_hull(card_image, upper_left_rectangle)
        image_ur = get_convex_hull(card_image, upper_right_rectangle)
        image_bl = get_convex_hull(card_image, bottom_left_rectangle)
        image_br = get_convex_hull(card_image, bottom_right_rectangle)

        hulls = [image_ul, image_ur, image_bl, image_br]

        result = cv2.drawContours(card_image, hulls, -1, (255, 0, 0), 1)
        cv2.imshow('result', result)
        cv2.waitKey()


if __name__ == '__main__':
    main()
