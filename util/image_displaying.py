import cv2


def display_image_and_wait(image, label='image'):
    cv2.imshow(label, image)
    cv2.waitKey()
