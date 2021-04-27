import cv2
import numpy as np
import utils

stream = True
path = 'sampleTable.jpg'
cap = cv2.VideoCapture('red_ball_1.mp4')
cap.set(3, 608)  # width
cap.set(4, 1080)  # height
wTable = 54  # inches
hTable = 108  # inches
scale = 4

warpMatrix = np.zeros((3, 3))
curWeight = 0

while cap.isOpened():
    if stream:
        success, img = cap.read()
    else:
        success, img = True, cv2.imread(path)

    if not success:
        print("Can't receive frame. Exiting...")
        break

    img, contours = utils.getContours(img, showCanny=True, draw=True)
    if len(contours) != 0:
        contourMax = contours[0]
        approxCorners = contourMax[2]
        imgWarp, newMatrix = utils.warpImg(img, approxCorners, wTable * scale, hTable * scale,
                                           prevMatrix=warpMatrix, prevWeight=curWeight)
        warpMatrix = newMatrix
        curWeight += 1
        cv2.imshow('Warped Table', imgWarp)

    img = cv2.resize(img, (0, 0), None, 0.7, 0.7)
    cv2.imshow('Original', img)

    k = cv2.waitKey(25) or 0xff
    if k == ord('q') or k == 27:
        break
    if k == ord('p') or k == 32:  # pause/play the video
        while True:
            key2 = cv2.waitKey(25) or 0xff
            cv2.imshow('Original', img)

            if key2 == ord('p') or k == 32:
                break
        # cv2.waitKey(-1)  # wait until any key is pressed

cap.release()
cv2.destroyAllWindows()


