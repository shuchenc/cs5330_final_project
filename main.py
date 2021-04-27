import cv2
import numpy as np
import utils
import detection

stream = True

path = 'sampleTable.jpg'
# path = 'youtube_game.png'
cap = cv2.VideoCapture('four_colors_1.mp4')
# cap = cv2.VideoCapture('red_ball_1.mp4')
# cap = cv2.VideoCapture('white_red_yellow_1.mp4')
# cap = cv2.VideoCapture('drill_fast.mp4')


wCap, hCap = 608, 1080
cap.set(3, 608)  # set width
cap.set(4, 1080)  # set height

wTable = 54  # inches
hTable = 108  # inches
scale = 4
wBirdseye = wTable * scale  # pixel
hBirdseye = hTable * scale  # pixel

warpMatrix = np.eye(3)
curMatrixWeight = 0

# The following two lists will be used to record the points to be drawn on each window
birdseyeToDraw = []
originalToDraw = []
toDrawLists = [originalToDraw, birdseyeToDraw]

birdseyeLineColor = (200, 0, 200)
originalLineColor = (200, 200, 0)

mouseX = -1
mouseY = -1

redPath, yellowPath, whitePath, greenPath = [], [], [], []
ballPathsToDraw = [redPath, yellowPath, whitePath, greenPath]
ballPathsColors = [(0, 0, 200), (0, 200, 200), (240, 240, 240), (160, 250, 250)]


def printXY(event, x, y, flags, param):
    global mouseY, mouseX, toDrawLists
    if event == cv2.EVENT_LBUTTONDOWN:
        # on left button down, add current mouse coordinates to the target list
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
imgRaw = np.zeros((wCap, hCap))
imgWarp = np.zeros((wBirdseye, hBirdseye))
debugMode = False
paused = False

while cap.isOpened():
    if not paused:
        if stream:
            success, imgRaw = cap.read()
        else:
            success = True
            imgRaw = cv2.imread(path)

        if not success:
            print("Can't receive frame. Exiting...")
            break

        frame_counter += 1
        # If the last frame is reached, reset the capture and the frame_counter
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0  # Or whatever as long as it is the same as next line
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for dl in toDrawLists:
                dl.clear()
            for bl in ballPathsToDraw:
                bl.clear()
        imgRaw = cv2.resize(imgRaw, (0, 0), None, 0.7, 0.7)

        deskArea = None
        img, contours = utils.getContours(imgRaw, showCanny=debugMode, draw=debugMode)

        if len(contours) != 0:
            contourMax = contours[0]
            approxCorners = contourMax[2]
            deskArea = approxCorners
            imgWarp, newMatrix = utils.warpImg(img, approxCorners, wBirdseye, hBirdseye,
                                               prevMatrix=warpMatrix, prevWeight=curMatrixWeight)

            img, redPos = detection.detectBall(img, deskArea, 'red')
            img, yellowPos = detection.detectBall(img, deskArea, 'yellow')
            img, whitePos = detection.detectBall(img, deskArea, 'white')
            img, greenPos = detection.detectBall(img, deskArea, 'green')
            ballPositions = [redPos, yellowPos, whitePos, greenPos]
            for i in range(len(ballPositions)):
                if ballPositions[i] is not None:
                    ballPathsToDraw[i].append(np.array([list(ballPositions[i])]))

            warpMatrix = newMatrix
            curMatrixWeight += 1
    else:
        img = imgRaw

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
    for j, ballPathToDraw in enumerate(ballPathsToDraw):
        if len(ballPathToDraw) >= 2:
            for i in range(len(ballPathToDraw) - 1):
                cv2.line(img, utils.np2tuple(ballPathToDraw[i]), utils.np2tuple(ballPathToDraw[i + 1]),
                         color=ballPathsColors[j], thickness=3)
                utils.drawWarpedLines(ballPathToDraw[i], ballPathToDraw[i + 1], warpMatrix, imgWarp,
                                      color=ballPathsColors[j])

    cv2.imshow('Warped Table', imgWarp)
    cv2.imshow('Original', img)

    k = cv2.waitKey(3) or 0xff
    if k == ord('q') or k == 27:
        break
    elif k == ord('p'):  # pause/play the video but allow drawing
        # while True:
        #     key2 = cv2.waitKey(25) or 0xff
        #     cv2.imshow('Original', img)
        #
        #     if key2 == ord('p') or k == 32:
        #         break
        #     elif k == ord('a'):
        #         print(mouseX, mouseY)
        paused = not paused
    elif k == 32:
        cv2.waitKey(-1)  # wait until any key is pressed
    elif k == ord('a'):
        print(mouseX, mouseY)
    elif k == ord('d'):
        debugMode = not debugMode
    elif k == ord('c'):
        originalToDraw.clear()
    elif k == ord('v'):
        birdseyeToDraw.clear()
    elif k == ord('b'):
        for ballPath in ballPathsToDraw:
            ballPath.clear()

cap.release()
cv2.destroyAllWindows()


