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
    # print("convex hull result: ", len(hull), hull)
    # peri = cv2.arcLength(hull, True)
    # approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
    # approx = reorder4Corners(approx)
    # print("convex approx result: ", len(approx), approx)

    # the following algo is more robust than cv2.approxPolyDP()
    approx = np.zeros_like(hull)[:4]
    points = hull.reshape((len(hull), 2))
    add = points.sum(1)
    approx[0] = points[np.argmin(add)]
    approx[2] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    approx[1] = points[np.argmin(diff)]
    approx[3] = points[np.argmax(diff)]

    return approx


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
    imgDilation = cv2.dilate(imgCanny, kernel=kernel, iterations=2)
    imgThresh = cv2.erode(imgDilation, kernel=kernel, iterations=1)

    # lower_red = np.array([110, 150, 50])
    # upper_red = np.array([180, 255, 180])
    # imgRed = filterColor(img, lower_red, upper_red)
    # imgRed = cv2.cvtColor(imgRed, cv2.COLOR_BGR2GRAY)
    # retV, imgRed = cv2.threshold(imgRed, 20, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Red', imgRed)
    # circles = cv2.HoughCircles(imgRed, cv2.HOUGH_GRADIENT, 1, 100)
    # print(circles)
    # circles = circles[0, :, :]
    # circles = np.uint16(np.around(circles))
    # for c in circles[:]:
    #     cv2.circle(img, (c[0], c[1]), c[2], (0, 255, 0), 5)
    #     cv2.circle(img, (c[0], c[1]), 2, (0, 255, 0), 10)

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
            # for i, p in enumerate(con[2]):
            #     curP = np2tuple(con[2])
            #     prevP = np2tuple(con[2])
            #     cv2.drawMarker(img, curP, (0, 200, 0), markerType=cv2.MARKER_TRIANGLE_DOWN)
            #     cv2.line(img, curP, prevP, (0, 255, 0), thickness=3)
            cv2.polylines(img, [con[2]], True, (0, 255, 0), thickness=2)
            # # draw hull points
            # for p in con[5]:
            #     cv2.drawMarker(img, (p[0, 0], p[0, 1]), (20, 200, 200), markerType=cv2.MARKER_TILTED_CROSS)
    if showCanny:
        cv2.imshow('Blue surface', imgBlue)
        cv2.imshow('Canny after dilation and erosion', imgThresh)
    return img, finalContours


def reorder4Corners(points):
    pointsNew = np.zeros_like(points)
    points = points.reshape((len(points), 2))
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[2] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[3] = points[np.argmax(diff)]
    return pointsNew


def warpImg(img, points, w, h, prevMatrix=np.zeros((3, 3)), prevWeight=0):
    assert(len(points == 4))
    reorderedPoints = reorder4Corners(points)
    pts1 = np.float32(reorderedPoints)
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    if prevMatrix.any():
        newMatrix = matrix * 1/(1+prevWeight) + prevMatrix * prevWeight/(1+prevWeight)
    else:
        newMatrix = matrix
    imgWarp = cv2.warpPerspective(img, newMatrix, (w, h))
    return imgWarp, newMatrix


def transferPts(pts, M):
    # pts should have a shape of n x 1 x 2
    homgPts = np.array([[x, y, 1] for [[x, y]] in pts]).T
    warpedPts = M.dot(homgPts)
    warpedPts /= warpedPts[2]
    warpedPts = np.array([[[round(x), round(y)]] for [x, y, _] in warpedPts.T])
    return warpedPts


def drawWarpedLines(pt1, pt2, M, dst, color=(0, 0, 255), thickness=2):
    [wp1, wp2] = transferPts([pt1, pt2], M)
    #print(wp1, wp2, wp1.shape, wp2.shape)
    wp1 = np2tuple(wp1)
    wp2 = np2tuple(wp2)
    cv2.line(dst, wp1, wp2, color, thickness)


def np2tuple(npPt):
    return tuple(npPt[0, :])

