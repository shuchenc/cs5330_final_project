import cv2
import numpy as np
from collections import deque

path = 'test-1.png'
# path = 'greenball.png'

# 初始化追踪点的列表
mybuffer = 8

# 设定阈值，HSV空间
balls = {
    'desk': {'range': [np.array([97, 121, 0]), np.array([180, 255, 255])], 'pts': deque(maxlen=mybuffer)},
    'red': {'range': [np.array([150, 150, 50]), np.array([180, 255, 180])], 'pts': deque(maxlen=mybuffer)},
    'yellow': {'range': [np.array([18, 109, 0]), np.array([40, 255, 254])], 'pts': deque(maxlen=mybuffer)},
    'white': {'range': [np.array([18, 0, 0]), np.array([120, 117, 255])], 'pts': deque(maxlen=mybuffer)},
    # following colors not working, too much noisy
    'green': {'range': [np.array([75, 100, 36]), np.array([96, 255, 255])], 'pts': deque(maxlen=mybuffer)},
    'blue': {'range': [np.array([107, 174, 0]), np.array([135, 255, 84])], 'pts': deque(maxlen=mybuffer)},
    'pink': {'range': [np.array([114, 36, 148]), np.array([179, 149, 255])], 'pts': deque(maxlen=mybuffer)},
}


def detectBalls(img, deskArea, color='red'):
    img = cv2.medianBlur(img, 5)
    cv2.imshow('medianblur', img)
    diameterBall = 30
    radiusBall = int(round(diameterBall / 2))
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2 * radiusBall * 0.9,
                               minRadius=int(round(radiusBall * 0.8)),
                               maxRadius=int(round(radiusBall * 1.2)),
                               param1=60,
                               param2=20
                               )
    print(circles)


def detectBall(frame, deskArea, color='red', showMask=False):
    # color space
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, balls[color]['range'][0], balls[color]['range'][1])
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # fill the background black to focus on poll table
    height, width = frame.shape[:2]
    tl = deskArea[0][0]
    tr = deskArea[1][0]
    br = deskArea[2][0]
    bl = deskArea[3][0]
    bgArea = np.array([[0, 0], [width, 0],
                       [width, height], [0, height],
                       [bl[0], bl[1]],
                       [br[0], br[1]], [tr[0], tr[1]], [tl[0], tl[1]],
                       [bl[0], bl[1]], [0, height]
                       ])
    cv2.fillPoly(mask, [bgArea], (0, 0, 0))
    if showMask:
        cv2.imshow('mask', mask)

    # output = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('output', output)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    # 轮廓检测
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # 初始化瓶盖圆形轮廓质心
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # for multiple object in same color
        for cnt in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            M = cv2.moments(cnt)
            if radius > 4 and M["m00"] > 0:
                c0 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.rectangle(frame, (c0[0] - 10, c0[1] - 10), (c0[0] + 10, c0[1] + 10), (0, 255, 0), 1)

        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # only proceed if the radius meets a minimum size
        M = cv2.moments(c)
        if radius > 3 and M["m00"] > 0:
            # 计算质心
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # determine if the point is in the desk area
            result = cv2.pointPolygonTest(deskArea, (center[0], center[1]), False)

            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            # cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # cv2.circle(img, center, 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (center[0] - 10, center[1] - 10), (center[0] + 10, center[1] + 10), (0, 255, 0), 1)
            # update the points queue
            balls[color]['pts'].appendleft(center)

        # 遍历追踪点，分段画出轨迹
        for i in range(1, len(balls[color]['pts'])):
            if balls[color]['pts'][i - 1] is None or balls[color]['pts'][i] is None:
                continue
            # 计算所画小线段的粗细
            thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)
            # thickness = 1
            # 画出小线段
            cv2.line(frame, balls[color]['pts'][i - 1], balls[color]['pts'][i], (0, 0, 255), thickness)

    return frame, center


# Load in image
# image = cv2.imread('test-1.png')
# detectDesk(image)
# while 1:
#     # detectDesk(image)
#     # Wait longer to prevent freeze for videos.
#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
