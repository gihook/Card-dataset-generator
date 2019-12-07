import numpy as np

card_width = 57
card_height = 87
corner_x_min = 2
corner_x_max = 10.5
corner_y_min = 2.5
corner_y_max = 23

zoom = 4
card_width *= zoom
card_height *= zoom
corner_x_min = int(corner_x_min * zoom)
corner_x_max = int(corner_x_max * zoom)
corner_y_min = int(corner_y_min * zoom)
corner_y_max = int(corner_y_max * zoom)

card_suits = ['s', 'h', 'd', 'c']
card_values = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

image_width = 720
image_height = 720

ref_card = np.array([[0, 0], [card_width, 0], [card_width, card_height], [0, card_height]], dtype = np.float32)
ref_card_rotated = np.array([[card_width, 0], [card_width, card_height], [0, card_height], [0, 0]], dtype = np.float32)
