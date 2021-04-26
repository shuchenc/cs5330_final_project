import cv2
import numpy as np


def filterColor(img, lower, upper, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    if debug:
        cv2.imshow('Color filtered image', res)
    return res

'''
Process the img and return a list of found contours
method based on Canny edge detection and cv2.findContours()
'''
def getContours(img, cThr=[100, 300], showCanny=False, minArea=1000, draw=False):
    lowerBlue = np.array([100, 100, 0], np.uint8)
    upperBlue = np.array([140, 250, 255], np.uint8)
    imgBlue = filterColor(img, lowerBlue, upperBlue)
    imgBlur = cv2.GaussianBlur(imgBlue, (25, 25), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDilation = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
    imgThresh = cv2.erode(imgDilation, kernel=kernel, iterations=2)
    if showCanny:
        cv2.imshow('Canny', imgThresh)

    img2, contours, hierachy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.03*peri, True)
            bbox = cv2.boundingRect(approx)
            finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
            for p in con[2]:
                cv2.drawMarker(img, (p[0, 0], p[0, 1]), (0, 200, 0), markerType=cv2.MARKER_TRIANGLE_DOWN)
    return img, finalContours


def reorder4Corners(points):
    pointsNew = np.zeros_like(points)
    points = points.reshape((len(points), 2))
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew


def warpImg(img, points, w, h):
    reorderedPoints = reorder4Corners(points)
    pts1 = np.float32(reorderedPoints)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp
