import numpy as np
import cv2


def prepare_image(img):
    height, width = img.shape[:2]
    half = int(img.shape[1] / 2)

    return img[:, half:, :]


def resize_image(img):
    resize_scale = 25
    width = int(img.shape[1] * resize_scale / 100)
    height = int(img.shape[0] * resize_scale / 100)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def display_image_and_wait(image, label='image'):
    cv2.imshow(label, image)
    cv2.waitKey()


def merge_images(images):
    return np.concatenate(images, axis=1)


def crop_rectangle(img, rect):
    center, size, angle = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[:2]

    rectangle_width, rectangle_height = size

    if rectangle_width > rectangle_height:
        angle += 90
        size = (size[1], size[0])

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

    cropped_image = cv2.getRectSubPix(rotated_image, size, center)

    return cropped_image, rotated_image


def process_file(card_img_path, debug=False):
    img = cv2.imread(card_img_path, cv2.IMREAD_COLOR)

    halfed_image = prepare_image(img)

    gray = cv2.cvtColor(halfed_image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    rectangle = cv2.minAreaRect(sorted_contours[0])
    angle = rectangle[2]
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)

    img_with_whitened_card = cv2.drawContours(gray.copy(), sorted_contours, 0,
                                              (255, 255, 255), cv2.FILLED)
    _, better_mask = cv2.threshold(img_with_whitened_card, 254, 255,
                                   cv2.THRESH_BINARY)
    better_masked_image = cv2.bitwise_and(halfed_image.copy(),
                                          halfed_image.copy(),
                                          mask=better_mask)

    image_with_contours = cv2.drawContours(halfed_image.copy(),
                                           sorted_contours, 0, (0, 255, 0), 3)
    image_with_contours = cv2.drawContours(image_with_contours.copy(), [box],
                                           0, (0, 0, 255), 3)

    cropped, rotated = crop_rectangle(better_masked_image, rectangle)

    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, alpha_mask = cv2.threshold(gray_cropped, 1, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(cropped)
    alphachannel = np.ones(b.shape, dtype=b.dtype) * 255
    alphachannel = cv2.bitwise_and(alphachannel.copy(),
                                   alphachannel.copy(),
                                   mask=alpha_mask)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    cropped[:, :, 3] = alphachannel

    if debug:
        alphachannel_4 = cv2.cvtColor(alphachannel, cv2.COLOR_GRAY2BGRA)
        display_image_and_wait(merge_images([cropped, alphachannel_4]))

        gray_3_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        better_mask_3_channel = cv2.cvtColor(better_mask, cv2.COLOR_GRAY2BGR)
        debug_image = merge_images([
            image_with_contours, gray_3_channel, better_mask_3_channel,
            better_masked_image
        ])
        debug_image = resize_image(debug_image)
        display_image_and_wait(debug_image)

    return cropped
