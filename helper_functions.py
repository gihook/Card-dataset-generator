import cv2

def resize_image(image):
    height, width, _ = image.shape
    img_scale = 650/width
    new_x, new_y = image.shape[1] * img_scale, image.shape[0] * img_scale
    resized_image = cv2.resize(image, (int(new_x), int(new_y)))

    return resized_image

def draw_image(container_name, image):
    resized_image = resize_image(image)
    cv2.imshow(container_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

