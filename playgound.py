import cv2
from models.scene import Scene
from util.image_displaying import display_image_and_wait, display_scene
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
from imgaug import augmenters as iaa
from util.file_listing import card_filenames
import random
from util.image_generating import prepare_background


def two_cards_scene(first: Scene, second: Scene):
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={
            "x": 25,
            "y": 10
        }, rotate=11)])
    new_image, new_bbs = seq(image=second.image, bounding_boxes=second.bbs)
    new_card = Scene.from_image(new_image, new_bbs)

    return merge(first, new_card)


def is_blocked(scene: Scene, bbx: BoundingBox):
    x = bbx.center_x
    y = bbx.center_y
    pixel = scene.image[int(y)][int(x)]

    return (len(list(filter(lambda x: x != 0, pixel))) > 0)


def merge(background: Scene, overlay: Scene):
    _, _, _, alphachannel = cv2.split(overlay.image)
    alphachannel = 255 - alphachannel
    result = cv2.bitwise_or(background.image.copy(),
                            background.image.copy(),
                            mask=alphachannel)
    result = cv2.bitwise_or(result, overlay.image)
    bbs = list(
        filter(lambda bbx: not is_blocked(overlay, bbx),
               background.bbs.bounding_boxes))
    bbs = bbs + overlay.bbs.bounding_boxes
    bbs = BoundingBoxesOnImage(bbs, background.image.shape)

    return Scene.from_image(result, bbs)


def random_transformation(scene: Scene):
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.3, 0.8),
                   translate_percent={
                       "x": (-0.1, 0.1),
                       "y": (-0.1, 0.1)
                   },
                   rotate=(-180, 180))
    ])
    new_image, new_bbs = seq(image=scene.image, bounding_boxes=scene.bbs)

    return Scene.from_image(new_image, new_bbs)


resized_cards_folder_name = "resized_images_with_alphachannel/"
filenames = card_filenames(resized_cards_folder_name)
cards = list(map(lambda f: Scene.from_path(f), filenames))


def get_random_card():
    index = random.randint(0, 51)

    return cards[index]


def read_background_scene(path):
    image = cv2.imread(path)
    image = prepare_background(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    print(image.shape)

    return Scene.from_image(image, BoundingBoxesOnImage([], image.shape))


background_path = "sarme.jpg"
background = read_background_scene(background_path)

for i in range(0, 50):
    first_card = get_random_card()
    second_card = get_random_card()
    third_card = get_random_card()
    scene = two_cards_scene(first_card, second_card)
    result = two_cards_scene(third_card, scene)
    # result = random_transformation(result)

    final = merge(background, result)
    display_scene(final)
