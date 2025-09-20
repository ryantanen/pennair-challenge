import cv2
import time
image = "dynamic.mp4"
video = cv2.VideoCapture(image)
def apply_filters(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Canny
    gray_image = cv2.Canny(gray_image, 25, 150)
    gray_image = cv2.dilate(gray_image, None, iterations=6)
    gray_image = cv2.erode(gray_image, None, iterations=6)
    thresh = cv2.threshold(gray_image, 150, 220, cv2.THRESH_BINARY)[1]

    return thresh

def find_contours(frame):
    contours, hierarchy = cv2.findContours(
        image=frame, 
        mode=cv2.RETR_TREE, 
        method=cv2.CHAIN_APPROX_NONE
    )
    return contours

def filter_contour(contour) -> bool:
    return cv2.contourArea(contour) > 500 and cv2.contourArea(contour) < 100000

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
    if cv2.waitKey(1) & 0xFF == ord('s'):
        showing_full_frame = not showing_full_frame
    cv2.imshow("Frame", draw_contours(frame if showing_full_frame else grey_scale, filtered_contours))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.01)

video.release()
cv2.destroyAllWindows()