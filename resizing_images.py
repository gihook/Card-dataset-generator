import numpy as np
import cv2
import os
from process_image import display_image_and_wait

zoom = 5

card_height = 88
card_width = 58

card_height *= zoom
card_width *= zoom

cards_folder = "card_images/"
card_filenames = os.listdir(cards_folder)
card_filenames = filter(lambda x: x != ".gitkeep", card_filenames)
card_filenames = sorted(card_filenames)
pwd = os.getcwd()

card_corners = [
    [0, 0],
    [card_width, 0],
    [card_width, card_height],
    [0, card_height],
]
ref_card = np.array(card_corners, dtype=np.float32)


def get_image_box(image):
    height, width = image.shape[:2]
    image_corners = [
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ]

    return np.array(image_corners, dtype=np.float32)


folder_name = "resized_images_with_alphachannel/"

for filename in card_filenames:
    image_path = os.path.join(pwd, cards_folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_box = get_image_box(image)
    transformation_matrix = cv2.getPerspectiveTransform(image_box, ref_card)
    image_warp = cv2.warpPerspective(image, transformation_matrix,
                                     (card_width, card_height))

    path = os.path.join(folder_name, '{filename}'.format(filename=filename))
    cv2.imwrite(path, image_warp)
    print(filename)
