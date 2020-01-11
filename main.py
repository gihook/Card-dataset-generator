import numpy as np
import cv2

# preparation: resizing and taking only green part
def prepare_image(img):
  half = int(img.shape[1] / 2)
  return img[:, half:, :]

def resize_image(img):
  resize_scale = 20
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

  contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

  rectangle = cv2.minAreaRect(sorted_contours[0])
  box = cv2.boxPoints(rectangle)
  box = np.int0(box)

  img_with_whitened_card = cv2.drawContours(gray.copy(), sorted_contours, 0, (255, 255, 255), cv2.FILLED)
  _, better_mask = cv2.threshold(img_with_whitened_card, 254, 255, cv2.THRESH_BINARY)
  better_masked_image = cv2.bitwise_and(img.copy(), img.copy(), mask = better_mask)
  
  img = cv2.drawContours(img.copy(), sorted_contours, 0, (0, 255, 0), 3)
  img = cv2.drawContours(img.copy(), [box], 0, (0, 0, 255), 3)
  
  return merge_images([img, gray_3_channel, better_masked_image])

def main():
  image_paths = ["./10h.jpg", "./2d.jpg", "./Jh.jpg", "./Kc.jpg", "./Kd.jpg"]
  
  for image_path in image_paths:
    result = process_file(image_path)
  
    resized_result = resize_image(result)
    display_image_and_wait(resized_result)

main()
