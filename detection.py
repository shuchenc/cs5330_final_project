import cv2
import numpy as np
from collections import deque

path = 'test-1.png'
# path = 'greenball.png'

import time

# 初始化追踪点的列表
mybuffer = 8

# 设定阈值，HSV空间
balls = {
    'desk':    {'range': [np.array([97, 121, 0]), np.array([180, 255, 255])], 'pts': deque(maxlen=mybuffer)},
    'red':     {'range': [np.array([150, 150, 50]), np.array([180, 255, 180])], 'pts': deque(maxlen=mybuffer)},
    'green':   {'range': [np.array([29, 86, 6]), np.array([64, 255, 255])], 'pts': deque(maxlen=mybuffer)},
    'white':   {'range': [np.array([27, 0, 120]), np.array([100, 113, 250])], 'pts': deque(maxlen=mybuffer)},
    'blue':    {'range': [np.array([0, 66, 137]), np.array([114, 255, 241])], 'pts': deque(maxlen=mybuffer)},
    # 'white':    [np.array([0, 0, 0]), np.array([0, 0, 255])]
}


def detectDesk(frame, color='blue'):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, balls[color]['range'][0], balls[color]['range'][1])
    mask = cv2.dilate(mask, None, iterations=10)
    # mask = cv2.erode(mask, None, iterations=4)

    cv2.imshow('mask', mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    print("Number of Contours found = " + str(len(contours)))
    # if len(contours) > 0:
        # for cnt in contours:
        #     M = cv2.moments(cnt)
        #     # 对象的质心
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])
        #
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     rect = cv2.minAreaRect(cnt)
        #     # 矩形四个角点取整
        #     box = np.int0(cv2.boxPoints(rect))
        #     cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
        #     ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        #     (x, y, radius) = cv2.int0((x, y, radius))
        #     cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
        #
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        #
        # rect = cv2.minAreaRect(contours[0])
        # box = np.int0(cv2.boxPoints(rect))
        # print(rect)
        # for point in box:
        #     cv2.circle(frame, (int(point[0]), int(point[1])), int(3), (0, 255, 255), 2)

    cv2.imshow('Frame', frame)
    return frame


def detectBall(frame, color='red'):
    # color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, balls[color]['range'][0], balls[color]['range'][1])
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('res', res)
    cv2.imshow('mask', mask)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    # 轮廓检测
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # 初始化瓶盖圆形轮廓质心
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # 计算质心
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 3:
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

    return frame


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