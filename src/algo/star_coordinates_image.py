
from typing import List
from src.star.star import Star
import os
import cv2
import json
import numpy as np
from numpy.linalg import norm

class StarCoordinatesImage:

    def __init__(self,nameImage: str = None):
        self._nameImage = nameImage

    def get_nameImage(self) -> str:
        return self._nameImage

    def set_nameImage(self,nameImage: str = None) -> None:
        self._nameImage = nameImage

    def coordinates_image_to_json(self):
        img_name=self._nameImage
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        if len(image.shape) == 3:
            image_bright = np.average(norm(image, axis=2)) / np.sqrt(3)
        else:
            image_bright = np.average(image)

        threshold = np.interp(image_bright, [0, 255], [50, 400])

        # threshold
        th, th_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        # find contours
        contours = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # filter by area
        id = 0
        stars = []
        for kp in contours:
            r = cv2.contourArea(kp)
            r = int(r / 2)
            if 5 < r < 40:
                M = cv2.moments(kp)
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                b = round(image[y, x] / 255, 2)
                #cv2.circle(th_img, (x, y), 40, (255, 0, 0), 3)
                cv2.circle(th_img, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(th_img, f'{id}', (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                new_star = {"id": id, "x": x, "y": y, "r": r, "b": b}
                stars.append(new_star)
                id += 1
            elif 5 < r:
                M = cv2.moments(kp)
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                cv2.circle(th_img, (x, y), 10, (0, 0, 0), -1)


        filename_with_extension = os.path.basename(img_name)
        filename_without_extension = os.path.splitext(filename_with_extension)[0]

        # save images
        images = np.concatenate((image, th_img), axis=1)
        cv2.imwrite("image/answer/stars" + filename_without_extension + ".jpg", th_img)

        string1 = "image/json/stars" + filename_without_extension + ".json"
        with open(string1, "w") as f:
            json.dump(stars, f)


    def cordinates_image_to_obj(self) -> List[Star]:
        filename_with_extension = os.path.basename(self._nameImage)
        filename_without_extension = os.path.splitext(filename_with_extension)[0]
        string1 = "image/json/stars" + filename_without_extension + ".json"
        with open(string1, 'r') as f:
            data = json.load(f)
        return [(Star(ID=obj['id'], x=obj['x'], y=obj['y'], b=obj['b'], r=obj['r'])) for obj in data]
