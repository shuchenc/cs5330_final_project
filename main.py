import cv2
import numpy as np
import utils
import detection

stream = True

path = 'sampleTable.jpg'
path = 'youtube_game.png'
# cap = cv2.VideoCapture('all-balls.mp4')
# cap = cv2.VideoCapture('red_ball_1.mp4')
cap = cv2.VideoCapture('drill_fast.mp4')
cap.set(3, 608)  # set width
cap.set(4, 1080)  # set height

wTable = 54  # inches
hTable = 108  # inches
scale = 4
wBirdseye = wTable * scale  # pixel
hBirdseye = hTable * scale  # pixel

warpMatrix = np.zeros((3, 3))
curMatrixWeight = 0

# The following two lists will be used to record the points to be drawn on each window
birdseyeToDraw = []
originalToDraw = []
toDrawLists = [originalToDraw, birdseyeToDraw]

birdseyeLineColor = (0, 200, 200)
originalLineColor = (200, 200, 0)

mouseX = -1
mouseY = -1


def printXY(event, x, y, flags, param):
    global mouseY, mouseX, toDrawLists
    if event == cv2.EVENT_LBUTTONDOWN:
        # on left button down, add current mouse coords to the target list
        mouseX, mouseY = x, y
        toDrawLists[param].append(np.array([[x, y]]))
        # print(param, mouseX, mouseY, originalToDraw, birdseyeToDraw)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # on right click, delete the latest point in the target list
        if len(toDrawLists[param]) > 0:
            toDrawLists[param].pop()
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # on db right click, delete the first point in the target list
        if len(toDrawLists[param]) > 0:
            toDrawLists[param].pop(0)


cv2.namedWindow('Original')
cv2.setMouseCallback('Original', printXY, param=0)
cv2.namedWindow('Warped Table')
cv2.setMouseCallback('Warped Table', printXY, param=1)

frame_counter = 0
while cap.isOpened():
    if stream:
        success, img = cap.read()
    else:
        success = True
        img = cv2.imread(path)

    if not success:
        print("Can't receive frame. Exiting...")
        break


    frame_counter += 1
    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    img = cv2.resize(img, (0, 0), None, 0.7, 0.7)

    img, contours = utils.getContours(img, showCanny=True, draw=True)
    if len(contours) != 0:
        contourMax = contours[0]
        approxCorners = contourMax[2]
        imgWarp, newMatrix = utils.warpImg(img, approxCorners, wBirdseye, hBirdseye,
                                           prevMatrix=warpMatrix, prevWeight=curMatrixWeight)
        warpMatrix = newMatrix
        curMatrixWeight += 1
        # now go through all the points that needs to be drawn for both windows:
        if len(birdseyeToDraw) >= 2:
            for i in range(len(birdseyeToDraw) - 1):
                cv2.line(imgWarp, utils.np2tuple(birdseyeToDraw[i]), utils.np2tuple(birdseyeToDraw[i+1]),
                         birdseyeLineColor)
                utils.drawWarpedLines(birdseyeToDraw[i], birdseyeToDraw[i + 1],
                                      np.linalg.inv(warpMatrix), img, color=birdseyeLineColor, thickness=3)
        if len(originalToDraw) >= 2:
            for i in range(len(originalToDraw) - 1):
                cv2.line(img, utils.np2tuple(originalToDraw[i]), utils.np2tuple(originalToDraw[i+1]),
                         originalLineColor, thickness=3)
                utils.drawWarpedLines(originalToDraw[i], originalToDraw[i + 1], warpMatrix, imgWarp,
                                      color=originalLineColor)
        cv2.imshow('Warped Table', imgWarp)

    img = cv2.resize(img, (0, 0), None, 0.7, 0.7)

    img = detection.detectBall(img, 'red')
    img = detection.detectBall(img, 'yellow')
    # img = detection.detectBall(img, 'desk')

    cv2.imshow('Original', img)

    k = cv2.waitKey(25) or 0xff
    if k == ord('q') or k == 27:
        break
    elif k == ord('p') or k == 32:  # pause/play the video
        # while True:
        #     key2 = cv2.waitKey(25) or 0xff
        #     cv2.imshow('Original', img)
        #
        #     if key2 == ord('p') or k == 32:
        #         break
        #     elif k == ord('a'):
        #         print(mouseX, mouseY)
        cv2.waitKey(-1)  # wait until any key is pressed
    elif k == ord('a'):
        print(mouseX, mouseY)

cap.release()
cv2.destroyAllWindows()


