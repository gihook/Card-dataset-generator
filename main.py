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

# preparation: resizing and taking only green part
def prepare_image(img):
  half = int(img.shape[1] / 2)
  return img[:, half:, :]

def resize_image(img):
  resize_scale = 17
  width = int(img.shape[1] * resize_scale / 100)
  height = int(img.shape[0] * resize_scale / 100)
  dim = (width, height)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def display_image_and_wait(image):
  cv2.imshow('image', image)
  cv2.waitKey()

def merge_images(images):
  return np.concatenate(images, axis = 1)

def process_file(card_img_path):
  img = cv2.imread(card_img_path, cv2.IMREAD_COLOR)
  
  img = prepare_image(img)
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_3_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  
  _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
  mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  masked_image = cv2.bitwise_and(img, img, mask = mask)

  return merge_images([img, gray_3_channel, mask_3_channel, masked_image])

image_paths = ["./10h.jpg", "./2d.jpg", "./Jh.jpg", "./Kc.jpg", "./Kd.jpg"]

for image_path in image_paths:
  result = process_file(image_path)

  resized_result = resize_image(result)
  display_image_and_wait(resized_result)
