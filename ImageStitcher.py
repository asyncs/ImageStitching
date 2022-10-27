import cv2 as cv
import numpy as np
from scipy.spatial import distance
from skimage.feature import plot_matches
import matplotlib.pyplot as plt


class ImageStitcher:

    def __init__(self, image_array):
        self.images = image_array
        self.keypoints_array = []
        self.descriptors_array = []
        self.matches = []

    def feature_extraction(self):
        tmp_images = self.images
        sift_keypoints_array = []
        sift_descriptors_array = []

        sift = cv.SIFT_create()

        for img in tmp_images:
            #img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(img, None)
            sift_keypoints_array.append(keypoints)
            sift_descriptors_array.append(descriptors)

            img_keypoint = cv.drawKeypoints(img, keypoints, None)
            plt.imshow(cv.cvtColor(img_keypoint, cv.COLOR_BGR2RGB))
            plt.show()

        self.keypoints_array = sift_keypoints_array
        self.descriptors_array = sift_descriptors_array

    #Sorting logic is inspired from https://github.com/AhmedHisham1/ORB-feature-matching/blob/master/utils.py
    def feature_matching_brute(self):
        matches = []
        for pair_index in range(len(self.descriptors_array) - 1):
            image_a = self.images[pair_index]
            image_b = self.images[pair_index + 1]
            keypoints_a = self.keypoints_array[pair_index]
            keypoints_b = self.keypoints_array[pair_index + 1]
            descriptor_a = self.descriptors_array[pair_index]
            descriptor_b = self.descriptors_array[pair_index + 1]

            distances_a2b = distance.cdist(descriptor_a, descriptor_b, metric='hamming')

            index_a = np.arange(descriptor_a.shape[0])
            index_b = np.argmin(distances_a2b, axis=1)

            matches_b2a = np.argmin(distances_a2b, axis=0)
            cross_check = index_a == matches_b2a[index_b]  # len(mask) = len(d1)

            index_a = index_a[cross_check]
            index_b = index_b[cross_check]
            distances_a2b = distances_a2b[index_a, index_b]
            sorted_indexes = distances_a2b.argsort()

            match = np.column_stack((index_a[sorted_indexes], index_b[sorted_indexes]))
            matches.append(match)

            keypoints_a_loc = np.flip(cv.KeyPoint_convert(keypoints_a), 1)
            keypoints_b_loc = np.flip(cv.KeyPoint_convert(keypoints_b), 1)

            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1)

            plot_matches(ax, cv.cvtColor(image_a, cv.COLOR_BGR2RGB), cv.cvtColor(image_b, cv.COLOR_BGR2RGB),
                         keypoints_a_loc, keypoints_b_loc, match)
            plt.show()

        self.matches = matches










