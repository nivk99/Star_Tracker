import math
import random
from typing import List
import numpy as np
from src.algo.star_coordinates_image import StarCoordinatesImage
from src.star.star import Star
import os
import cv2
from itertools import combinations


class StarMatchTriangles:

    def __init__(self, name_image_star_database: str = None, name_image_star_frame: str = None):
        self._name_image_star_database = name_image_star_database
        self._name_image_star_frame = name_image_star_frame
        self._coordinates_star_frame: StarCoordinatesImage = None
        self._coordinates_star_database: StarCoordinatesImage = None
        self._list_star_database: List[Star] = None
        self._list_star_frame: List[Star] = None
        self._id_database={}
        self.id_frame={}
        self.__save_image_to_json()
        self.__json_to_obj()
        self.__id_frame_database()
        self.algo()

    def get_name_Image_star_frame(self) -> str:
        return self._name_image_star_frame

    def get_name_Image_star_database(self) -> str:
        return self._name_image_star_database

    def set_name_Image_star_frame(self, name_image_star_frame: str = None) -> None:
        self._name_image_star_frame = name_image_star_frame

    def set_name_Image_star_database(self, name_image_star_database: str = None) -> None:
        self._name_image_star_database = name_image_star_database

    def __save_image_to_json(self) -> None:
        self._coordinates_star_database = StarCoordinatesImage(self._name_image_star_database)
        self._coordinates_star_database.coordinates_image_to_json()

        self._coordinates_star_frame = StarCoordinatesImage(self._name_image_star_frame)
        self._coordinates_star_frame.coordinates_image_to_json()


    def __json_to_obj(self) -> None:
        self._list_star_database = self._coordinates_star_database.cordinates_image_to_obj()
        self._list_star_frame = self._coordinates_star_frame.cordinates_image_to_obj()

    def __id_frame_database(self) -> None:
        for k in self._list_star_database:
            self._id_database[k.get_id()]=k
        for k in self._list_star_frame:
            self.id_frame[k.get_id()]=k

    def algo(self) -> None:
        choice_list_database = self.__choice_from(self._list_star_database)
        distance_list_database = [self.__minimum_distance(s[0], s[1], s[2]) for s in choice_list_database]
        choice_list_frame = self.__choice_from(self._list_star_frame)
        distance_list_frame = [self.__minimum_distance(s[0], s[1], s[2]) for s in choice_list_frame]

        ans = {k.get_id(): [] for k in self._list_star_frame}
        ans_dic = {k.get_id(): 1000 for k in self._list_star_frame}

        for p in distance_list_frame:
            rms = self.__RMS(p, distance_list_database[0])
            ind_d1 = distance_list_database[0][0]
            ind_d2 = distance_list_database[0][1]
            ind_d3 = distance_list_database[0][2]
            ind_f1 = p[0]
            ind_f2 = p[1]
            ind_f3 = p[2]
            for s in distance_list_database:
                temp = self.__RMS(p, s)
                if temp < rms:
                    rms = temp
                    ind_d1, ind_d2, ind_d3 = s[0], s[1], s[2]
                    ind_f1, ind_f2, ind_f3 = p[0], p[1], p[2]
            rms_half = (ind_f1[0] - ind_d1[0]) ** 2 / 2
            if ans_dic[ind_f1[1].get_id()] > rms_half:
                ans_dic[ind_f1[1].get_id()] = rms_half
                ans[ind_f1[1].get_id()] = ind_d1[1]
            rms_half = (ind_f2[0] - ind_d2[0]) ** 2 / 2
            if ans_dic[ind_f2[1].get_id()] > rms_half:
                ans_dic[ind_f2[1].get_id()] = rms_half
                ans[ind_f2[1].get_id()] = ind_d2[1]
            rms_half = (ind_f3[0] - ind_d3[0]) ** 2 / 2
            if ans_dic[ind_f3[1].get_id()] > rms_half:
                ans_dic[ind_f3[1].get_id()] = rms_half
                ans[ind_f3[1].get_id()] = ind_d3[1]
        self.__eachPointDifferentColor(ans)

    def __minimum_distance(self, p1, p2, p3) -> list:
        distances = [
            (math.dist([p1.get_x(), p1.get_y()], [p2.get_x(), p2.get_y()]), p1),
            (math.dist([p2.get_x(), p2.get_y()], [p3.get_x(), p3.get_y()]), p2),
            (math.dist([p3.get_x(), p3.get_y()], [p1.get_x(), p1.get_y()]), p3)
        ]
        distances.sort(key=lambda x: x[0])  # Sort distances in ascending order
        return distances

    def __choice_from(self, arr):
        if len(arr) < 3:
            return arr
        else:
            return list(combinations(arr, 3))

    def __RMS(self, s, p):
        tol = 1e-6  # Tolerance for ratio comparison
        ratio1 = s[0][0] / p[0][0]
        ratio2 = s[1][0] / p[1][0]
        ratio3 = s[2][0] / p[2][0]
        if math.isclose(ratio1, ratio2, rel_tol=tol) and math.isclose(ratio1, ratio3, rel_tol=tol):
            return 0
        if math.isclose(ratio1, ratio3, rel_tol=tol) and math.isclose(ratio2, ratio2, rel_tol=tol):
            return 0
        if math.isclose(ratio2, ratio3, rel_tol=tol) and math.isclose(ratio1, ratio2, rel_tol=tol):
            return 0
        if math.isclose(ratio2, ratio3, rel_tol=tol) and math.isclose(ratio1, ratio1, rel_tol=tol):
            return 0
        if math.isclose(ratio1, ratio2, rel_tol=tol) and math.isclose(ratio1, ratio3, rel_tol=tol):
            return 0
        if math.isclose(ratio1, ratio3, rel_tol=tol) and math.isclose(ratio2, ratio1, rel_tol=tol):
            return 0
        if math.isclose(ratio2, ratio2, rel_tol=tol) and math.isclose(ratio1, ratio3, rel_tol=tol):
            return 0
        if math.isclose(ratio2, ratio1, rel_tol=tol) and math.isclose(ratio1, ratio2, rel_tol=tol):
            return 0
        if math.isclose(ratio3, ratio1, rel_tol=tol) and math.isclose(ratio1, ratio2, rel_tol=tol):
            return 0
        if math.isclose(ratio3, ratio2, rel_tol=tol) and math.isclose(ratio1, ratio3, rel_tol=tol):
            return 0
        return math.sqrt(((s[0][0] - p[0][0]) ** 2 + (s[1][0] - p[1][0]) ** 2 + (s[2][0] - p[2][0]) ** 2) / 3)

    def __eachPointDifferentColor(self,arr:{int:list}):
        filename_with_extension = os.path.basename(self._name_image_star_database)
        filename_without_extension1 = os.path.splitext(filename_with_extension)[0]

        filename_with_extension = os.path.basename(self._name_image_star_frame)
        filename_without_extension2 = os.path.splitext(filename_with_extension)[0]

        img1 = cv2.imread(self._name_image_star_database)
        img2 = cv2.imread(self._name_image_star_frame)
        blue_colors222 = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (128, 0, 128),
                          (0, 255, 255), (0, 0, 0), (255, 255, 255), (165, 42, 42), (255, 215, 0), (192, 192, 192),
                          (255, 36, 0), (178, 34, 34)]
        list_col=[]
        for i in arr:
            x2=self.id_frame[i].get_x()
            y2=self.id_frame[i].get_y()
            x1=arr[i].get_x()
            y1=arr[i].get_y()
            print(f"frame: {self.id_frame[i]} <----> database: {arr[i]}")

            c1=random.randint(0,255)
            c2=random.randint(0,255)
            c3=random.randint(0,255)
            col=(c1,c2,c3)
            while col in list_col:
                c1 = random.randint(0, 255)
                c2 = random.randint(0, 255)
                c3 = random.randint(0, 255)
                col = (c1, c2, c3)
            list_col.append(col)
            cv2.circle(img1, (x1, y1), radius=30, color=col, thickness=-1)
            cv2.circle(img2, (x2, y2), radius=30, color=col, thickness=-1)

        string1 = "image/answer/image_with_point_stars1" + filename_without_extension1 + ".jpg"
        cv2.imwrite(string1, img1)

        string1 = "image/answer/image_with_point_stars2" + filename_without_extension2 + ".jpg"
        cv2.imwrite(string1, img2)

    def calculate_angular_distance(self,side_a, side_b, side_c):
        """
        Calculates the angular distance (in degrees) in a non-right triangle using the Law of Cosines.

        Args:
            side_a (float): Length of side a.
            side_b (float): Length of side b.
            side_c (float): Length of side c.

        Returns:
            float: The angular distance in degrees.
        """
        # Calculate the cosine of the angle using the Law of Cosines
        cosine_angle = (side_a ** 2 + side_b ** 2 - side_c ** 2) / (2 * side_a * side_b)

        # Convert cosine to radians and then calculate the inverse cosine (arccosine)
        radians_angle = math.acos(cosine_angle)

        # Convert radians to degrees and return the angular distance
        degrees_angle = math.degrees(radians_angle)

        return degrees_angle