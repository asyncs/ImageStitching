import cv2 as cv
import numpy as np
from scipy.spatial import distance
from skimage.feature import plot_matches
import matplotlib.pyplot as plt


def get_cartesian_coordinates(homogeneous_array):
    return np.array(homogeneous_array[:-1] / homogeneous_array[-1])


def get_homogeneous_coordinates(cartesian_array):
    return np.array(cartesian_array + [1])


def perspective_projection(point_x, point_y, homography_matrix):
    hom_xy = get_homogeneous_coordinates([point_x, point_y])
    projected_point = np.matmul(homography_matrix, hom_xy)
    return projected_point


def find_inliers(points_loc_a, points_loc_b, matches, homography, threshold):
    inlier_count = 0
    inlier_set = []
    for every in matches:
        a_xy = points_loc_a[every[0]]
        b_xy = points_loc_b[every[1]]
        b_xy_hat = perspective_projection(a_xy[0], a_xy[1], homography)
        b_xy_hat = get_cartesian_coordinates(b_xy_hat)

        error = distance.pdist([b_xy, b_xy_hat], metric='euclidean')
        if error < threshold:
            inlier_count = inlier_count + 1
            inlier_set.append(every)

    return inlier_count, inlier_set


def fit_homography(points_a, points_b):
    pair_matrix = []

    for index in range(points_a.shape[0]):
        x_a = points_a[index, 0]
        y_a = points_a[index, 1]
        x_b = points_b[index, 0]
        y_b = points_b[index, 1]
        pair_matrix.append([-x_a, -y_a, -1, 0, 0, 0, x_a*x_b, y_a*x_b, x_b])
        pair_matrix.append([0, 0, 0, -x_a, -y_a, -1, x_a*y_b, y_a*y_b, y_b])

    pair_matrix = np.matrix(pair_matrix)

    _, _, vT = np.linalg.svd(pair_matrix)
    H = np.reshape(vT[8], (3, 3))
    H = (1 / H.item(8)) * H

    return np.array(H)


def ransac_fitting(matches, keypoint_a, keypoint_b, iter_count=500, epsilon=2):
    ransac_iteration = iter_count
    ransac_threshold = epsilon
    keypoints_a_loc = cv.KeyPoint_convert(keypoint_a)
    keypoints_b_loc = cv.KeyPoint_convert(keypoint_b)
    max_inlier_count = 0
    max_inlier_set = []

    for iter_index in range(ransac_iteration):
        random_matches = np.random.randint(0, matches.shape[0], size=4)
        points_a = []
        points_b = []

        for each in random_matches:
            points_a.append(keypoint_a[matches[each, 0]].pt)
            points_b.append(keypoint_b[matches[each, 1]].pt)
        homography = fit_homography(np.array(points_a), np.array(points_b))
        inlier_count, inlier_set = find_inliers(keypoints_a_loc, keypoints_b_loc, matches, homography, ransac_threshold)

        print("Iteration: ", iter_index+1, "Inlier Count: ", inlier_count)

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            max_inlier_set = inlier_set

    print("Maximum inlier count: ", max_inlier_count)
    return np.array(max_inlier_set)


class ImageStitcher:

    def __init__(self, image_array):
        self.images = image_array
        self.keypoints_array = []
        self.descriptors_array = []
        self.matches = []
        self.homography_matrices = []

    def feature_extraction(self):
        tmp_images = self.images
        sift_keypoints_array = []
        sift_descriptors_array = []

        sift = cv.SIFT_create()

        for img in tmp_images:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            sift_keypoints_array.append(keypoints)
            sift_descriptors_array.append(descriptors)

            img_keypoint = cv.drawKeypoints(img, keypoints, None)
            plt.imshow(cv.cvtColor(img_keypoint, cv.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Extracted Keypoints")
            plt.show()

        self.keypoints_array = sift_keypoints_array
        self.descriptors_array = sift_descriptors_array

    # Cross-check logic is inspired from https://github.com/AhmedHisham1/ORB-feature-matching/blob/master/utils.py
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
            cross_check = index_a == matches_b2a[index_b]

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
            plt.axis('off')
            plt.title("Matched Keypoints")
            plt.show()

        self.matches = matches

    def fitting_homography(self):
        homography_matrices = []
        for pair_index in range(len(self.descriptors_array) - 1):
            image_a = self.images[pair_index]
            image_b = self.images[pair_index + 1]
            keypoints_a = self.keypoints_array[pair_index]
            keypoints_b = self.keypoints_array[pair_index + 1]
            matches = self.matches[pair_index]

            inliers = ransac_fitting(matches, keypoints_a, keypoints_b)

            keypoints_a_loc = np.flip(cv.KeyPoint_convert(keypoints_a), 1)
            keypoints_b_loc = np.flip(cv.KeyPoint_convert(keypoints_b), 1)

            points_a = []
            points_b = []
            for each in inliers:
                points_a.append(keypoints_a[each[0]].pt)
                points_b.append(keypoints_b[each[1]].pt)

            homography = fit_homography(np.array(points_a), np.array(points_b))
            homography_matrices.append(homography)

            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1)

            plot_matches(ax, cv.cvtColor(image_a, cv.COLOR_BGR2RGB), cv.cvtColor(image_b, cv.COLOR_BGR2RGB),
                         keypoints_a_loc, keypoints_b_loc, inliers)
            plt.axis('off')
            plt.title("Matched Keypoints after RANSAC")
            plt.show()

        self.homography_matrices = homography_matrices

    def alignment(self):
        for pair_index in range(len(self.descriptors_array) - 1):
            image_a = self.images[pair_index]
            image_b = self.images[pair_index + 1]
            result = cv.warpPerspective(image_b, np.linalg.inv(self.homography_matrices[0]), (image_a.shape[1] + image_b.shape[1], image_a.shape[0]+image_b.shape[0]))

            result[0:image_a.shape[0], 0:image_a.shape[1]] = image_a

            plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Stitched Image")
            plt.show()


        pass
