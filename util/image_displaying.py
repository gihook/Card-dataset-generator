import cv2
from models.scene import Scene


def display_image_and_wait(image, label='image'):
    cv2.imshow(label, image)
    cv2.waitKey()


def display_scene(scene: Scene):
    image = cv2.cvtColor(scene.image, cv2.COLOR_BGRA2BGR)
    result = scene.bbs.draw_on_image(image, size=1, color=(244, 11, 8))
    display_image_and_wait(result)
