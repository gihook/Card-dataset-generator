import numpy as np
import cv2
import os

# preparation: taking only green part
def prepare_image(img):
  height, width = img.shape[:2]
  half = int(img.shape[1] / 2)
  return img[:, half:, :]

def resize_image(img):
  resize_scale = 25
  width = int(img.shape[1] * resize_scale / 100)
  height = int(img.shape[0] * resize_scale / 100)
  dim = (width, height)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def display_image_and_wait(image, label = 'image'):
  cv2.imshow(label, image)
  cv2.waitKey()

def merge_images(images):
  return np.concatenate(images, axis = 1)

def crop_rectangle(img, rect):
    center, size, angle = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[:2]

    rectangle_width, rectangle_height = size

    if rectangle_width > rectangle_height:
      angle += 90
      size = (size[1], size[0])

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

def process_file(card_img_path, debug = False):
  img = cv2.imread(card_img_path, cv2.IMREAD_COLOR)
  
  halfed_image = prepare_image(img)
  
  gray = cv2.cvtColor(halfed_image, cv2.COLOR_BGR2GRAY)
  
  _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

  contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

  rectangle = cv2.minAreaRect(sorted_contours[0])
  angle = rectangle[2]
  box = cv2.boxPoints(rectangle)
  box = np.int0(box)

  img_with_whitened_card = cv2.drawContours(gray.copy(), sorted_contours, 0, (255, 255, 255), cv2.FILLED)
  _, better_mask = cv2.threshold(img_with_whitened_card, 254, 255, cv2.THRESH_BINARY)
  better_masked_image = cv2.bitwise_and(halfed_image.copy(), halfed_image.copy(), mask = better_mask)
  
  image_with_contours = cv2.drawContours(halfed_image.copy(), sorted_contours, 0, (0, 255, 0), 3)
  image_with_contours = cv2.drawContours(image_with_contours.copy(), [box], 0, (0, 0, 255), 3)

  cropped, rotated = crop_rectangle(better_masked_image, rectangle)
  
  if debug:
    debug_image = merge_images([image_with_contours, better_masked_image])
    debug_image = resize_image(debug_image)
    display_image_and_wait(debug_image)

  return cropped

def main():
  folder_path = "input_images/"
  folder_files = os.listdir(folder_path)
  folder_files = sorted(folder_files)
  pwd = os.getcwd()
  image_paths = map(lambda file_name: (os.path.join(pwd, folder_path, file_name), file_name), folder_files)

  for (image_path, file_name) in image_paths:
    result = process_file(image_path)
    cv2.imwrite('card_images/{file_name}'.format(file_name = file_name) + ".jpg", result)
    print('saved file {file_name}'.format(file_name = file_name))

main()
