import cv2
import numpy as np
import utils
import detection

stream = True
path = 'sampleTable.jpg'
# cap = cv2.VideoCapture('all-balls.mp4')
cap = cv2.VideoCapture('red_ball_1.mp4')
cap.set(3, 608)  # width
cap.set(4, 1080)  # height
wTable = 54  # inches
hTable = 108  # inches
scale = 4

warpMatrix = np.zeros((3, 3))
curWeight = 0

frame_counter = 0
while cap.isOpened():
    if stream:
        success, img = cap.read()
    else:
        success, img = True, cv2.imread(path)

    if not success:
        print("Can't receive frame. Exiting...")
        break

    frame_counter += 1
    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

    img = detection.detectBall(img, 'red')
    # img = detection.detectBall(img, 'desk')

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


