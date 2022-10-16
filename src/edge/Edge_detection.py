"""
file containing the class that defines a simple approach for self driving car using
line detection with Hough transform
"""

import cv2
import numpy as np
from PIL import Image
from mss import mss
import os
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from shared.CONFIG import *
from sendInput import PressKey, ReleaseKey, Z, Q, D
from shared.utils import *


class EdgeDetection(object):
    def __init__(self):
        self.sct = mss()

    def get_image(self):
        countdown(Count_Down)
        print("Start EdgeDetection ...")
        print("Press 'q' or 'x' to finish...")
        zeros = np.array([[0, 0, 0, 0]])
        while 1:
            try:
                screenshot = self.sct.grab(mon)  # grab a screenShot of the screen
                img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                can = self.canny(img)  # apply canny filter
                roi = self.region_of_interest(can)  # focus on the region of interest
                lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 100, np.array([]), minLineLength=minLineLength,
                                        maxLineGap=maxLineGap)  # get all the lines in the img
                averaged_lines = self.average_slope_intercept(can, lines, threshold)  # get average line
                # if there are two lines we go straight
                if ((averaged_lines[0] == zeros).all() == False) and ((averaged_lines[1] == zeros).all() == False):
                    PressKey(Z)
                    ReleaseKey(Z)
                # if there is only the line on the right we go left
                if ((averaged_lines[0] == zeros).all() == True) and ((averaged_lines[1] == zeros).all() == False):
                    PressKey(Q)
                    ReleaseKey(Q)
                # if there is only the line on the left we go right
                if ((averaged_lines[0] == zeros).all() == False) and ((averaged_lines[1] == zeros).all() == True):
                    PressKey(D)
                    ReleaseKey(D)

                line_image = self.display_lines(img, averaged_lines)
                combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
                cv2.imshow('Screen', combo_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"error 1 : {e}")

    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        poly = np.array([[
            (0, height),
            (0, 2.5 * int(height / 3)),
            (int(width / 2), int(1.5 * height / 3)),
            (width, 2.5 * int(height / 3)),
            (width, height), ]], np.int32)
        cv2.fillPoly(mask, poly, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def canny(self, img):
        canny = cv2.Canny(img, 50, 110)
        return canny

    def average_slope_intercept(self, image, lines, threshold):
        left_fit = []
        right_fit = []
        if lines is None:
            return None
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:

                    fit = np.polyfit((x1, x2), (y1, y2), 1)
                    slope = fit[0]
                    intercept = fit[1]
                    if abs(slope) > threshold:
                        if slope < 0:  # y is reversed in image
                            # print(f"slope : {slope} , intercept : {intercept}")
                            left_fit.append((slope, intercept))
                        else:
                            right_fit.append((slope, intercept))

            # add more weight to longer lines
            if len(left_fit) == 0:
                left_fit_average = np.zeros(1)
                left_line = [[0, 0, 0, 0]]
            else:
                left_fit_average = np.average(left_fit, axis=0)
                left_line = self.make_points(image, left_fit_average)
            if len(right_fit) == 0:
                right_fit_average = np.zeros(1)
                right_line = [[0, 0, 0, 0]]
            else:
                right_fit_average = np.average(right_fit, axis=0)
                right_line = self.make_points(image, right_fit_average)
            averaged_lines = np.array([left_line, right_line])
            return averaged_lines
        except Exception as e:
            print(f"error 2: {e}")
            return None

    def make_points(self, image, line):
        slope, intercept = line
        y1 = int(image.shape[0])  # bottom of the image
        y2 = int(y1 * 3 / 5)  # slightly lower than the middle
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def display_lines(self, img, lines):
        line_image = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

if __name__ == '__main__':
    EdgeDetector = EdgeDetection()
    EdgeDetector.get_image()