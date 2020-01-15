import os
import cv2
from process_image import display_image_and_wait
from resizing_images import zoom

pwd = os.getcwd()
upper_left_rectangle = [(2 * zoom, 3 * zoom), (10 * zoom, 24 * zoom)]
upper_right_rectangle = [(47 * zoom, 3 * zoom), (55 * zoom, 24 * zoom)]
bottom_left_rectangle = [(2 * zoom, 64 * zoom), (10 * zoom, 84 * zoom)]
bottom_right_rectangle = [(47 * zoom, 64 * zoom), (55 * zoom, 84 * zoom)]

cards_folder = "resized_images_with_alphachannel/"

card_filenames = os.listdir(cards_folder)
card_filenames = filter(lambda x: x != ".gitkeep", card_filenames)
card_filenames = sorted(card_filenames)


def display_image_with_rectangles(image, points):
    display_image = image

    for point in points:
        p1, p2 = point
        display_image = cv2.rectangle(image,
                                      p1,
                                      p2,
                                      color=(0, 0, 255),
                                      thickness=1)
    display_image_and_wait(display_image)


def overlay_transparent(background_img,
                        img_to_overlay_t,
                        x,
                        y,
                        overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),
                              roi.copy(),
                              mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


background_image = cv2.imread("./sample-background.jpg")


def main():
    for filename in card_filenames:
        image_path = os.path.join(pwd, cards_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        display_image_with_rectangles(image, [
            upper_left_rectangle, upper_right_rectangle, bottom_left_rectangle,
            bottom_right_rectangle
        ])

        result_image = overlay_transparent(background_image, image, 30, 30)


if __name__ == '__main__':
    main()
