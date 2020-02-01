from models.scene import Scene
from util.image_displaying import display_image_and_wait

path = "resized_images_with_alphachannel/9s.png"
card = Scene.from_path(path)

display_image_and_wait(card.image)
