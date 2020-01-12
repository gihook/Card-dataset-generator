import os
import cv2
from process_image import process_file


def main():
    folder_path = "input_images/"
    folder_files = os.listdir(folder_path)
    folder_files = sorted(folder_files)
    pwd = os.getcwd()
    image_paths = map(
        lambda file_name:
        (os.path.join(pwd, folder_path, file_name), file_name), folder_files)

    for (image_path, file_name) in image_paths:
        result = process_file(image_path)
        filename = f'card_images/{file_name}.jpg'
        cv2.imwrite(filename, result)
        print(f'saved file {filename}')


if __name__ == '__main__':
    main()
