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
    H = (1 / H[2, 2]) * H

    return np.array(H)


def ransac_fitting(matches, keypoint_a, keypoint_b, iter_count=200, epsilon=2):
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

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            max_inlier_set = inlier_set

    return np.array(max_inlier_set)


def get_new_dimensions(image_a, image_b, homography):
    image_a_width = image_a.shape[1]
    image_a_height = image_a.shape[0]
    image_b_width = image_b.shape[1]
    image_b_height = image_b.shape[0]

    homography_inv = np.linalg.inv(homography)

    top_left_corner = get_cartesian_coordinates(perspective_projection(0, 0, homography_inv))
    bottom_left_corner = get_cartesian_coordinates(perspective_projection(0, image_b_height-1, homography_inv))
    top_right_corner = get_cartesian_coordinates(perspective_projection(image_b_width-1, 0, homography_inv))
    bottom_right_corner = get_cartesian_coordinates(perspective_projection(image_b_width-1, image_b_height-1, homography_inv))

    minimum_width = int(min([0, top_left_corner[0], bottom_left_corner[0]]))
    minimum_height = int(min([0, top_left_corner[1], top_right_corner[1]]))
    max_width = int(max([0, top_right_corner[0], bottom_right_corner[0]]))
    max_height = int(max([0, bottom_left_corner[1], bottom_right_corner[1]]))

    offset_width = np.abs(minimum_width)
    offset_height = np.abs(minimum_height)

    new_width = offset_width + image_a_width + np.abs(image_a_width - max_width)
    new_height = offset_height + image_a_height + np.abs(image_a_height - max_height)

    start_alpha = int(min([top_left_corner[0], bottom_left_corner[0]]))

    return new_width, new_height, offset_width, offset_height, minimum_width, minimum_height, start_alpha


def alpha_blending(image_anc, image_new, start, count):
    alpha = np.linspace(0, 1, count)
    blended_img = image_new.copy()

    for row in range(image_anc.shape[0]):
        for col in range(image_anc.shape[1]):
            if np.sum(image_new[row, col, :]) == 0:
                blended_img[row, col, :] = image_anc[row, col, :]
            elif np.sum(image_anc[row, col, :]) == 0:
                blended_img[row, col, :] = image_new[row, col, :]

    for col in range(alpha.shape[0]):
        for row in range(image_anc.shape[0]):
            if np.sum(image_new[row, start + col, :]) == 0:
                blended_img[row, start + col, :] = image_anc[row, start + col, :]
            elif np.sum(image_anc[row, start + col, :]) == 0:
                blended_img[row, start + col, :] = image_new[row, start + col, :]
            else:
                blended_img[row, start + col, :] = (1 - alpha[col]) * image_anc[row, start + col, :] + alpha[col] * image_new[row, start + col, :]

    return blended_img


def average_conv(image):
    kernel_avg = np.ones((3, 3), np.float32) / 9
    average_blurring = cv.filter2D(src=image, ddepth=-1, kernel=kernel_avg)
    return average_blurring


def stitch_images(image_a, image_b, homography):
    new_width, new_height, offset_width, offset_height, min_width, min_height, start_alpha = get_new_dimensions(image_a, image_b, homography)

    new_image_a = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_image_b = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for x in range(image_a.shape[1]):
        for y in range(image_a.shape[0]):
            new_image_a[y+offset_height, x+offset_width, :] = image_a[y, x, :]

    for x in range(min_width, new_width):
        for y in range(min_height, new_height):
            x_start = x + offset_width
            y_start = y + offset_height
            if x_start < new_width and y_start < new_height:
                x_new, y_new = get_cartesian_coordinates(perspective_projection(x, y, homography))
                if 0 < x_new < image_b.shape[1] and 0 < y_new < image_b.shape[0]:
                    x_new = int(x_new)
                    y_new = int(y_new)
                    new_image_b[y_start, x_start, :] = image_b[y_new, x_new, :]

    count = image_a.shape[1] - start_alpha

    new_image = alpha_blending(new_image_a, new_image_b, start_alpha, count)

    return new_image


def feature_extraction(images):
    sift_keypoints_array = []
    sift_descriptors_array = []

    sift = cv.SIFT_create()

    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        sift_keypoints_array.append(keypoints)
        sift_descriptors_array.append(descriptors)

        image_keypoint = cv.drawKeypoints(image, keypoints, None)

        plt.imshow(cv.cvtColor(image_keypoint, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Extracted Keypoints")
        plt.show()

    return sift_descriptors_array, sift_keypoints_array


# Cross-check logic is inspired from https://github.com/AhmedHisham1/ORB-feature-matching/blob/master/utils.py
def feature_matching_brute(images, descriptors_array, keypoints_array):
    matches = []
    for pair_index in range(len(descriptors_array) - 1):
        image_a = images[pair_index]
        image_b = images[pair_index + 1]
        keypoints_a = keypoints_array[pair_index]
        keypoints_b = keypoints_array[pair_index + 1]
        descriptor_a = descriptors_array[pair_index]
        descriptor_b = descriptors_array[pair_index + 1]

        distances_a2b = distance.cdist(descriptor_a, descriptor_b, metric='euclidean')

        matches_a2b = np.argmin(distances_a2b, axis=1)
        matches_b2a = np.argmin(distances_a2b, axis=0)

        cross_check = np.arange(descriptor_a.shape[0]) == matches_b2a[matches_a2b]

        index_a = np.arange(descriptor_a.shape[0])[cross_check]
        index_b = matches_a2b[cross_check]

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

    return matches


def find_homography(images, keypoints_array, matches, epsilon):
    homography_matrix = None
    for pair_index in range(len(keypoints_array) - 1):
        image_a = images[pair_index]
        image_b = images[pair_index + 1]
        keypoints_a = keypoints_array[pair_index]
        keypoints_b = keypoints_array[pair_index + 1]
        matches = matches[pair_index]

        inliers = ransac_fitting(matches, keypoints_a, keypoints_b, epsilon=epsilon)

        points_a = []
        points_b = []
        for each in inliers:
            points_a.append(keypoints_a[each[0]].pt)
            points_b.append(keypoints_b[each[1]].pt)

        homography = fit_homography(np.array(points_a), np.array(points_b))
        homography_matrix = homography

        keypoints_a_loc = np.flip(cv.KeyPoint_convert(keypoints_a), 1)
        keypoints_b_loc = np.flip(cv.KeyPoint_convert(keypoints_b), 1)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)

        plot_matches(ax, cv.cvtColor(image_a, cv.COLOR_BGR2RGB), cv.cvtColor(image_b, cv.COLOR_BGR2RGB),
                     keypoints_a_loc, keypoints_b_loc, inliers)
        plt.axis('off')
        plt.title("Matched Keypoints after RANSAC")
        plt.show()

    return homography_matrix


def panorama(image_a, image_b, epsilon):
    sift_descriptors_array, sift_keypoints_array = feature_extraction([image_a, image_b])
    matches = feature_matching_brute([image_a, image_b], sift_descriptors_array, sift_keypoints_array)
    homography_matrix = find_homography([image_a, image_b], sift_keypoints_array, matches, epsilon=epsilon)

    result = stitch_images(image_a, image_b, homography_matrix)

    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Stitched Image")
    plt.show()

    return result


class ImageStitcher:

    def __init__(self, image_array, epsilon):
        self.images = image_array
        self.epsilon = epsilon

    def alignment(self):
        image_old = None
        for pair_index in range(len(self.images) - 1):
            if pair_index == 0:
                image_a = self.images[pair_index]
                image_b = self.images[pair_index + 1]

                image_old = panorama(image_a, image_b, self.epsilon)
            else:
                image_a = image_old
                image_b = self.images[pair_index + 1]

                image_old = panorama(image_a, image_b, self.epsilon)
