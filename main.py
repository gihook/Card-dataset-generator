import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pickle
from glob import glob
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon

from helper_functions import resize_image, draw_image
from global_variables import ref_card, card_width, card_height

card_img_path = "qc.jpg"

img = cv2.imread(card_img_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 11, 17)
edge = cv2.Canny(gray, 30, 200)

contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

index = 0
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

contour_color = (0, 255, 0)
contour_thickness = 3
# cv2.drawContours(img, sorted_contours, index, contour_color, contour_thickness)

selected_contour = sorted_contours[0]
rectangle = cv2.minAreaRect(selected_contour)
box = cv2.boxPoints(rectangle)
box = np.int0(box)
# image = cv2.drawContours(img, [box], 0, (0, 0, 255), 5)

new_perspective = cv2.getPerspectiveTransform(np.float32(box), ref_card)
card_image = cv2.warpPerspective(img, new_perspective, (card_width, card_height))

print(rectangle)
print(ref_card)

draw_image('img', img)
draw_image('card', card_image)
