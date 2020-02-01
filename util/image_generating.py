import numpy as np

generated_image_height = 608
generated_image_width = 608


def empty_background():
    return np.zeros((generated_image_height, generated_image_width, 4),
                    dtype=np.uint8)


def card_with_empty_backgroud(card_image):
    card_height, card_width = card_image.shape[:2]

    image_center_x = int(generated_image_width / 2)
    image_center_y = int(generated_image_height / 2)

    card_center_x = int(card_width / 2)
    card_center_y = int(card_height / 2)

    empty_image = empty_background()
    empty_image[(image_center_y - card_center_y):(image_center_y +
                                                  card_height - card_center_y),
                (image_center_x - card_center_x):(card_width + image_center_x -
                                                  card_center_x)] = card_image

    return empty_image
