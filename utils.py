import cv2
import numpy as np


def filterColor(img, lower, upper, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    if debug:
        cv2.imshow('Color filtered image', res)
    return res


# returns the approximated four corners of a (trapezoid) contour
def getCornerApprox(contour):
    hull = cv2.convexHull(contour)
    # peri = cv2.arcLength(hull, True)
    # print("convex hull result: ", len(hull), hull)
    # approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
    # print("convex approx result: ", len(approx), approx)

    # the following algo is more robust than cv2.approxPolyDP()
    newApprox = np.zeros_like(hull)[:4]
    points = hull.reshape((len(hull), 2))
    add = points.sum(1)
    newApprox[0] = points[np.argmin(add)]
    newApprox[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newApprox[1] = points[np.argmin(diff)]
    newApprox[2] = points[np.argmax(diff)]

    return newApprox


'''
Process the img and return a list of found contours
method based on Canny edge detection and cv2.findContours()
'''
def getContours(img, cThr=(100, 300), showCanny=False, minArea=1000, draw=False):
    lowerBlue = np.array([100, 100, 0], np.uint8)
    upperBlue = np.array([140, 250, 255], np.uint8)
    imgBlue = filterColor(img, lowerBlue, upperBlue)
    imgBlur = cv2.GaussianBlur(imgBlue, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((10, 10))
    imgDilation = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
    imgThresh = cv2.erode(imgDilation, kernel=kernel, iterations=3)

    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    finalContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            hull = cv2.convexHull(contour)
            approx = getCornerApprox(contour)
            bbox = cv2.boundingRect(approx)
            finalContours.append([len(approx), area, approx, bbox, contour, hull])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[5], -1, (0, 0, 255), 3)
            for p in con[2]:
                cv2.drawMarker(img, (p[0, 0], p[0, 1]), (0, 200, 0), markerType=cv2.MARKER_TRIANGLE_DOWN)
            # # draw hull points
            # for p in con[5]:
            #     cv2.drawMarker(img, (p[0, 0], p[0, 1]), (20, 200, 200), markerType=cv2.MARKER_TILTED_CROSS)
    if showCanny:
        cv2.imshow('Canny', imgThresh)
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


def warpImg(img, points, w, h, prevMatrix=np.zeros((3, 3)), prevWeight=0):
    assert(len(points == 4))
    reorderedPoints = reorder4Corners(points)
    pts1 = np.float32(reorderedPoints)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    if prevMatrix.any():
        newMatrix = matrix * 1/(1+prevWeight) + prevMatrix * prevWeight/(1+prevWeight)
    else:
        newMatrix = matrix
    imgWarp = cv2.warpPerspective(img, newMatrix, (w, h))
    return imgWarp, newMatrix
