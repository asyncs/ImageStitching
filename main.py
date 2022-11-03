import os
import cv2 as cv

from ImageStitcher import ImageStitcher


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_path = 'data_image_stitching/'

    images = []
    sub_images = []
    prev_image_index = '1'

    for image in os.listdir(img_path):
        image_name = os.fsdecode(image)
        if prev_image_index != image_name[2]:
            images.append(sub_images)
            sub_images = []
            prev_image_index = image_name[2]

        img = cv.imread(img_path + image_name)
        sub_images.append(img)

    images.append(sub_images)

    stitcher1 = ImageStitcher(images[0], epsilon=5)
    stitcher1.alignment()
    stitcher2 = ImageStitcher(images[1], epsilon=2.5)
    stitcher2.alignment()
    stitcher3 = ImageStitcher(images[2], epsilon=2)
    stitcher3.alignment()
