import cv2.xfeatures2d as cv
from matplotlib import pyplot as plt
import random
import numpy as np
from mpmath import norm
import csv
import os
import cv2


# https://github.com/ayushgarg31/Feature-Matching

class StarMatchSiftKnnBbs:

    def __init__(self, img_name_1, img_name_2):
        self._img_name_2 = img_name_2
        self._img_name_1 = img_name_1
        self.SIFT_KNN_BBS()

    def get_img_name_1(self):
        return self._img_name_1

    def get_img_name_2(self):
        return self._img_name_2

    def img(self, img_name):
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        if len(image.shape) == 3:
            image_bright1 = np.average(norm(image, axis=2)) / np.sqrt(3)
        else:
            image_bright1 = np.average(image)

        threshold = np.interp(image_bright1, [0, 255], [50, 400])

        # threshold
        th, th_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # find contours
        contours = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # filter by area
        id = 0
        for kp in contours:
            r = cv2.contourArea(kp)
            r = int(r / 2)
            if 5 < r < 40:
                M = cv2.moments(kp)
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                cv2.circle(th_img, (x, y), 10, (0, 0, 255), 2)
                id += 1
            elif 40 < r:
                M = cv2.moments(kp)
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                cv2.circle(th_img, (x, y), 10, (0, 0, 0), -1)

        return th_img

    # Define a function called SIFT_KNN_BBS which takes two image file paths as input (img1 and img2):
    def SIFT_KNN_BBS(self):
        # Read the input images in grayscale:
        t1 = self.img(self._img_name_1)
        t2 = self.img(self._img_name_2)

        # Create a SIFT object:
        sift = cv.SIFT_create()

        # Detect keypoints and compute descriptors for both images:
        kp1, des1 = sift.detectAndCompute(t1, None)
        kp2, des2 = sift.detectAndCompute(t2, None)

        # Draw keypoints on the input images:
        f = cv2.drawKeypoints(t1, kp1, None, [0, 0, 255], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        nf = cv2.drawKeypoints(t2, kp2, None, [255, 0, 0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Create a BFMatcher object:
        bf = cv2.BFMatcher()
        # Perform KNN matching on the descriptors of both images:
        matches = bf.knnMatch(des1, des2, k=2)

        # Filter the matches using a ratio test and store the good matches:
        good1 = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good1.append([m])

        # Perform KNN matching on the descriptors of the second image with the first image:
        matches = bf.knnMatch(des2, des1, k=2)

        # Filter the matches using a ratio test and store the good matches:
        good2 = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good2.append([m])

        # Further filter the matches by checking if the corresponding keypoints are the same in both images, and store the good matches:
        good = []

        for i in good1:
            img1_id1 = i[0].queryIdx
            img2_id1 = i[0].trainIdx

            (x1, y1) = kp1[img1_id1].pt
            (x2, y2) = kp2[img2_id1].pt

            for j in good2:
                img1_id2 = j[0].queryIdx
                img2_id2 = j[0].trainIdx

                (a1, b1) = kp2[img1_id2].pt
                (a2, b2) = kp1[img2_id2].pt

                if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                    good.append(i)

        # Extract matched keypoints and calculate displacement, radius, and brightness
        tracked_stars = []
        for i, match in enumerate(good):
            # Generate a random color for visualization
            color = tuple([random.randint(0, 255) / 255.0 for _ in range(3)])
            img1_id = match[0].queryIdx
            img2_id = match[0].trainIdx
            (x1, y1) = kp1[img1_id].pt
            (x2, y2) = kp2[img2_id].pt
            tracked_stars.append({
                'x': x1,
                'y': y1,
                'dx': x2,
                'dy': y2,
                'color': color,
                'id': i
            })

        # Plot matched keypoints on images with unique colors for visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(t1, cv2.COLOR_BGR2RGB))
        plt.title('Image 1')
        for match in tracked_stars:
            plt.plot(match['x'], match['y'], 'o', markersize=5, color=match['color'])
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(t2, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        for match in tracked_stars:
            plt.plot(match['dx'], match['dy'], 'o', markersize=5, color=match['color'])

        filename_with_extension = os.path.basename(self._img_name_1)
        filename_without_extension1 = os.path.splitext(filename_with_extension)[0]
        filename_with_extension = os.path.basename(self._img_name_2)
        filename_without_extension2 = os.path.splitext(filename_with_extension)[0]

        # Save the plot as an image file
        plt.savefig(f'image/answer/{filename_without_extension1}_{filename_without_extension2}.png', dpi=300,
                    bbox_inches='tight')
        plt.show()
        # Close the plot
        plt.close()

        # Save the good matches to a CSV file
        with open(f'image/answer/{filename_without_extension1}_{filename_without_extension2}.csv', 'w',
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'color', 'x1', 'y1', 'x2', 'y2'])
            for match in tracked_stars:
                writer.writerow([match['id'], match['color'], match['x'], match['y'], match['dx'], match['dy']])
