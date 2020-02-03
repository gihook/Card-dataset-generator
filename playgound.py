import cv2
from models.scene import Scene
from util.image_displaying import display_image_and_wait, display_scene
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
from imgaug import augmenters as iaa
from util.file_listing import card_filenames
import random
from util.image_generating import prepare_background
from image_augmentation import write_image, write_to_txt_file, random_string
from image_augmentation import get_paths_and_names, folder_path, classes, get_random_background

image_paths = get_paths_and_names(folder_path)


def two_cards_scene(first: Scene, second: Scene):
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={
            "x": 25,
            "y": 10
        }, rotate=11)])
    new_image, new_bbs = seq(image=second.image, bounding_boxes=second.bbs)
    new_card = Scene.from_image(new_image, new_bbs)

    return first.merge(new_card)


resized_cards_folder_name = "resized_images_with_alphachannel/"
filenames = card_filenames(resized_cards_folder_name)
cards = list(map(lambda f: Scene.from_path(f), filenames))


def get_random_card():
    index = random.randint(0, 51)

    return cards[index]


def read_background_scene(image):
    image = prepare_background(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    return Scene.from_image(image, BoundingBoxesOnImage([], image.shape))


def random_scene(num):
    first = get_random_card()
    second = get_random_card()
    third = get_random_card()
    scene = two_cards_scene(first, second)
    scene = two_cards_scene(third, scene).random_transformation()

    for i in range(3, num):
        random_card = get_random_card().random_transformation()
        scene = scene.merge(random_card)

    return scene


for i in range(0, 55000):
    print("iteration:", i)
    result = random_scene(6)
    background_image = get_random_background()

    background = read_background_scene(background_image)
    final = background.merge(result)

    file_name = random_string()
    write_to_txt_file(file_name, final.bbs.bounding_boxes)
    write_image(file_name, final.image)
    # display_scene(final)
