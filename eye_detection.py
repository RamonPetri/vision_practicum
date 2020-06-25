import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load image
image_1 = cv2.imread("free-profile-photo-whatsapp-4.png", 0)
image_2 = cv2.imread("pexels-photo-220453.jpeg", 0)


cv2.waitKey(1000)

def histogram(image):
    image = cv2.medianBlur(image, 5)
    hist = cv2.calcHist(image, [0], None, [256], [0, 256])
    return hist.all()


def circle_detection(image):
    circle_hist = []

    # Preprocess
    image = cv2.medianBlur(image, 5)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Go looking for circles
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=44, param2=37, maxRadius=0, minRadius=0)
    # Put found circle points into array
    circles = np.uint16(np.around(circles))

    # Draw real circles on these points
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(color_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(color_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Test", color_image)
    cv2.waitKey(1000)

    for i in circles[0, :]:
        circle_hist.append(histogram(i))

    return circle_hist


if circle_detection(image_1) == circle_detection(image_2):
    print(True)

