import cv2 as cv
import numpy as np
import math
cap = cv.VideoCapture(0)

def adjust_gamma(image, gamma=1.0):
	
	table = np.array([((i / 255.0) ** gamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv.LUT(image, table)


def histo_equalization(image: cv.Mat):
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    l, a, b = cv.split(lab_image)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    equalized_l = clahe.apply(l)

    equalized_lab_image = cv.merge([equalized_l, a, b])

    return cv.cvtColor(equalized_lab_image, cv.COLOR_LAB2BGR)



def resize_frame(frame: cv.Mat, output_width: int, output_height: int):
    original_height, original_width = frame.shape[:2]
    width_scale = output_width / original_width
    height_scale = output_height / original_height
    return cv.resize(frame, (0, 0), fx=min(width_scale, height_scale), fy=min(width_scale, height_scale))


def getMask(hsv: cv.Mat, lower_range, upper_range):
    lower_range_hsv = np.array(lower_range)
    upper_range_hsv = np.array(upper_range)
    return cv.inRange(hsv, lower_range_hsv, upper_range_hsv)


red_lines_threshold = 30
canvas = None
prev_point, current_point = None, None
initialized = False
waiting_frames = 0

height = 720
width = 1280
_,frame = cap.read()
frame = resize_frame(frame,width,height)
height,width = frame.shape[:2]


screenshot_old_distance = 500
screenshot_dicreasing = False
capturing = False
screenshot_blink = 0
screenshots=0

zoom_old_distance = 500
zoom_dicreasing = False
zoom_amount = 1
zoom_ratio = 0.2


while True:
    _, frame = cap.read()
    frame = resize_frame(frame, width, height)

    # flip the frame horizontally
    frame = cv.flip(frame, 1) 

    if not initialized:
        canvas = np.zeros_like(frame)
        initialized = True

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    

    mask_red_light = getMask(hsv, [0,150,120], [10,255,255])

    mask1_red_dark = getMask(hsv, [170,150,120], [180,255,255])

    mask_red = mask_red_light+mask1_red_dark


    
    mask_green = getMask(hsv, [82, 122, 65],[90, 255, 127]) + getMask(hsv,[30, 50, 40], [90, 255, 100])
    
    mask_blue = getMask(hsv,[100, 80, 80],[120, 255, 255])
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    mask_green = cv.erode(mask_green, kernel, iterations=2)
    mask_green = cv.dilate(mask_green, kernel, iterations=6)

    mask_red = cv.erode(mask_red, kernel, iterations=2)
    mask_red = cv.dilate(mask_red, kernel, iterations=2)

    mask_blue = cv.erode(mask_blue, kernel, iterations=2)
    mask_blue = cv.dilate(mask_blue, kernel, iterations=2)


    # frame = cv.bitwise_and(frame, frame, mask=mask_green)

    red_contours, _ = cv.findContours(mask_red.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv.findContours(mask_green.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    green_contours = tuple(filter(lambda x: cv.contourArea(x) > 1000, green_contours))

    blue_contours, _ = cv.findContours(mask_blue.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blue_contours = tuple(filter(lambda x: cv.contourArea(x) > 1000, blue_contours))


    cv.rectangle(frame, (0,0), (100, 100), (55,55,55), -1)
    cv.putText(frame, "CLEAR", (10, 55), cv.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), thickness=2)
    clearBtn = np.array([[0,0],[0,100], [100, 100] , [100,0]])
    clear = -1

    if len(red_contours) > 0:
        fingertip_contour = max(red_contours, key=cv.contourArea)
        if cv.contourArea(fingertip_contour) > 1000:
            M = cv.moments(fingertip_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_point = (cx, cy)
                if current_point:
                    clear = cv.pointPolygonTest(clearBtn, current_point, False)
                if clear > 0:
                    canvas = np.zeros_like(frame)
                else:
                    waiting_frames = 0
    else:
        waiting_frames += 1

    if waiting_frames > red_lines_threshold:
        prev_point = None
        current_point = None
        waiting_frames = 0
            
    
    if prev_point == None:
        prev_point = current_point

    if current_point and prev_point != current_point and clear < 0:
        cv.line(canvas, prev_point, current_point, (0, 0, 255), 3)
        prev_point = current_point

        
    frame = cv.addWeighted(frame, 0.9, canvas, 1.2, 0)
#------------------------------------------------------------------- Resize
    if len(green_contours) > 1:
        green_contours = sorted(green_contours, key=cv.contourArea, reverse=True)
        green_fingertip_contour1 = green_contours[0]
        green_fingertip_contour2 = green_contours[1]
        if cv.contourArea(green_fingertip_contour1) > 1000 and cv.contourArea(green_fingertip_contour2) > 1000:
            M1 = cv.moments(green_fingertip_contour1)
            current_point1 = (0,0)
            current_point2 = (0,0)
            if M1["m00"] > 0:
                cx = int(M1["m10"] / M1["m00"])
                cy = int(M1["m01"] / M1["m00"])
                current_point1 = (cx,cy)
            M2 = cv.moments(green_fingertip_contour2)
            if M2["m00"] > 0:
                cx = int(M2["m10"] / M2["m00"])
                cy = int(M2["m01"] / M2["m00"])
                current_point2 = (cx,cy)
            distance = math.dist(current_point1,current_point2)
            distance = max(0,distance - 100)
            if zoom_old_distance == -1:
                zoom_old_distance=distance
            zoom_amount += zoom_ratio*(distance-zoom_old_distance)/100
            # zoom_amount = 1 + zoom_ratio*(distance)/100 # according to distance between fingers
            zoom_amount = max(1,zoom_amount)
            zoom_old_distance = distance
    else:
        zoom_old_distance=-1
#--------------------------------------------------- Screenshot
    if len(blue_contours) > 1:
        blue_contours = sorted(blue_contours, key=cv.contourArea, reverse=True)
        blue_fingertip_contour1 = blue_contours[0]
        blue_fingertip_contour2 = blue_contours[1]
        if cv.contourArea(blue_fingertip_contour1) > 1000 and cv.contourArea(blue_fingertip_contour2) > 1000:
            M1 = cv.moments(blue_fingertip_contour1)
            current_point1 = (0,0)
            current_point2 = (0,0)
            if M1["m00"] > 0:
                cx = int(M1["m10"] / M1["m00"])
                cy = int(M1["m01"] / M1["m00"])
                current_point1 = (cx,cy)
            M2 = cv.moments(blue_fingertip_contour2)
            if M2["m00"] > 0:
                cx = int(M2["m10"] / M2["m00"])
                cy = int(M2["m01"] / M2["m00"])
                current_point2 = (cx,cy)
            distance = math.dist(current_point1,current_point2)
            if distance < screenshot_old_distance:
                screenshot_dicreasing = True
                print("dicrease " + str(distance))
            elif distance > 100:
                screenshot_dicreasing = False              
                capturing = False
                print("increase " + str(distance))
            if distance < 100 and screenshot_dicreasing and capturing == False:
                filename = "screenshots/screenshot" + str(screenshots) + ".jpg"
                print("Captured! " + str(screenshots))
                cv.imwrite(filename,frame)
                capturing = True
                screenshots += 1
                screenshot_blink=3
            screenshot_old_distance = distance
    elif len(blue_contours) > 0 and screenshot_dicreasing and capturing == False and screenshot_old_distance < 100:
        filename = "screenshots/screenshot" + str(screenshots) + ".jpg"
        print("Captured! " + str(screenshots))
        cv.imwrite(filename,frame)
        capturing = True
        screenshots += 1
        screenshot_blink=3
    
    if screenshot_blink > 0:
        frame = np.full(frame.shape, 255, dtype = np.uint8)
        screenshot_blink -= 1
    
    # Here resize
    new_height = height*zoom_amount
    new_width = width*zoom_amount
    frame = resize_frame(frame,new_width,new_height)

    # Here crop
    height_gaurd = int((new_height-height)/2.0)
    width_gaurd = int((new_width-width)/2.0)
    frame = frame[height_gaurd:height_gaurd+height,width_gaurd:width_gaurd+width]
    cv.imshow("Req1", frame)

    

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()