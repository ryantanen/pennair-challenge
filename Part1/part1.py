import cv2


image = cv2.imread("static.png")

cv2.imshow("Image", image)
def apply_filters(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Canny
    gray_image = cv2.Canny(gray_image, 25, 150)
    gray_image = cv2.dilate(gray_image, None, iterations=3)
    gray_image = cv2.erode(gray_image, None, iterations=3)
    thresh = cv2.threshold(gray_image, 150, 220, cv2.THRESH_BINARY)[1]

    return thresh

thresh = apply_filters(image)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
print(hierarchy)

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
            cv2.putText(img=image_copy, text="Center", org=above_center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1)
            cv2.drawMarker(img=image_copy, position=center, color=(0,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
            # Label coords below center
            # Offset by 50 pixels below center
            below_center = (center[0] - 50 , center[1] + 50)
            cv2.putText(img=image_copy, text=str(center), org=below_center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)
    return filtered_contours

cv2.startWindowThread()
image_copy = image.copy()
image_copy = cv2.drawContours(image_copy, process_contours(contours), -1, (0, 0, 255), 2)
showing_full_frame = False

while True:
    cv2.imshow("Image", thresh if showing_full_frame else image_copy)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        showing_full_frame = not showing_full_frame
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()