import cv2
import numpy as np


def region_of_interest(edges):

    height, width = edges.shape

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (0, int(height*0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped = cv2.bitwise_and(edges, mask)

    return cropped


def detect_lanes(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        1,
        np.pi/180,
        threshold=80,
        minLineLength=80,
        maxLineGap=30
    )

    if lines is not None:

        for line in lines:

            x1,y1,x2,y2 = line[0]

            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),4)

    return frame