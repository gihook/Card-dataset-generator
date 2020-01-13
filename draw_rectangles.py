import os
import cv2
from process_image import display_image_and_wait

zoom = 5
pwd = os.getcwd()
upper_left_rectangle = [(2 * zoom, 3 * zoom), (10 * zoom, 24 * zoom)]

cards_folder = "resized_images_with_alphachannel/"

card_filenames = os.listdir(cards_folder)
card_filenames = filter(lambda x: x != ".gitkeep", card_filenames)
card_filenames = sorted(card_filenames)


def display_image_with_rectangles(image, point):
    p1, p2 = point
    display_image = cv2.rectangle(image,
                                  p1,
                                  p2,
                                  color=(0, 0, 255),
                                  thickness=1)
    display_image_and_wait(display_image)


for filename in card_filenames:
    image_path = os.path.join(pwd, cards_folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    display_image_with_rectangles(image, upper_left_rectangle)
