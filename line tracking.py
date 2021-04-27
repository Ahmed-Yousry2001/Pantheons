import cv2
import numpy as np

s = 1

video = cv2.VideoCapture('line tracking.mp4')

#frame = cv2.imread("1.jpg")
while True:
    ret, img = video.read()

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_blue = np.array([92, 50, 50])
    upper_blue = np.array([120, 255, 255])

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    #lines = cv2.HoughLinesP(thresh, 2,np.pi/180,100,minLineLength=210,maxLineGap=20)
    cnts = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (36, 255, 12), 2)

#    if s == 1:
#        if lines is not None:
            #continue
#            for line in lines:
#                x1, y1, x2, y2 = line[0]
#                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#/
    cv2.imshow('mask', mask)
    cv2.imshow("image", img)
    key = cv2.waitKey(10)
    if key == 27:
        break
        video.release()
        cv2.destroyAllWindows()