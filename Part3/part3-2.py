import cv2
import time
image = "dynamic2.mp4"
video = cv2.VideoCapture(image)

backSub_KNN = cv2.createBackgroundSubtractorKNN(detectShadows=False)
backSub_MOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

def apply_filters(frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgMask = backSub_KNN.apply(frame)
    # fgMask = backSub_MOG2.apply(frame)
    # fgMask = cv2.GaussianBlur(fgMask, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    # dilate and erode

    return fgMask

def find_contours(frame):
    contours, hierarchy = cv2.findContours(
        image=frame, 
        mode=cv2.RETR_TREE, 
        method=cv2.CHAIN_APPROX_NONE
    )
    return contours

def filter_contour(contour) -> bool:
    min_area = 500
    area = cv2.contourArea(contour)
    
    # Filter by area
    if area < min_area or area > 100000:
        return False
    
    # Filter by aspect ratio to get more regular shapes
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
        return False
    
    # Filter by extent (ratio of contour area to bounding rectangle area)
    rect_area = w * h
    extent = float(area) / rect_area
    if extent < 0.5:  # Too irregular
        return False
    
    # Filter by solidity (ratio of contour area to convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = float(area) / hull_area
        if solidity < 0.3:  # Too concave/irregular
            return False
    
    return True

def get_contour_center(contour) -> tuple:
    M = cv2.moments(contour)
    # Centriods 
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def process_contours(_contours):
    filtered_contours = []
    for _contour in _contours:
        if filter_contour(_contour):
            filtered_contours.append(_contour)
            center = get_contour_center(_contour)
            # Center marker
            above_center = (center[0] - 20, center[1] - 20)
            cv2.putText(img=frame, text="Center", org=above_center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1)
            cv2.drawMarker(img=frame, position=center, color=(0,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
            # Label coords below center
            # Offset by 50 pixels below center
            below_center = (center[0] - 50 , center[1] + 50)
            cv2.putText(img=frame, text=str(center), org=below_center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)
    return filtered_contours

def draw_contours(frame, contours):
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    return frame

showing_full_frame = False
while True:
    ret, frame = video.read()
    if not ret:
        break
    grey_scale = apply_filters(frame)
    contours = find_contours(grey_scale)
    filtered_contours = process_contours(contours)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        showing_full_frame = not showing_full_frame
    elif key == ord('q'):
        break
        
    cv2.imshow("Frame", draw_contours(frame if showing_full_frame else grey_scale, filtered_contours))

video.release()
cv2.destroyAllWindows()