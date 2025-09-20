import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading
from collections import deque
image = "dynamic2.mp4"
video = cv2.VideoCapture(image)

# Camera intrinsic matrix
K = np.array([[2564.3186869, 0, 0],
              [0, 2569.70273111, 0],
              [0, 0, 1]], dtype=np.float32)

# Camera parameters
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]  # Will be set based on image center
cy = K[1, 2]  # Will be set based on image center

backSub_KNN = cv2.createBackgroundSubtractorKNN(detectShadows=False)
backSub_MOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# 3D tracking data structures
object_3d_positions = deque(maxlen=100)  # Store recent 3D positions
depth_estimator = None  # Will initialize depth estimation model

def estimate_depth_monocular(frame, contour):
    """
    Estimate depth using monocular cues like object size and position
    This is a simplified approach - in practice you'd use neural networks
    """
    # Get contour properties
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Simple depth estimation based on object size and vertical position
    # Assume objects lower in frame are closer (ground plane assumption)
    height, width = frame.shape[:2]
    
    # Normalize y position (0 = top, 1 = bottom)
    y_norm = y / height
    
    # Base depth estimation (in meters, adjust based on your scene)
    # Objects at bottom of frame are closer
    base_depth = 2.0 + (1.0 - y_norm) * 8.0  # 2-10 meters range
    
    # Adjust depth based on object size (larger objects are typically closer)
    size_factor = np.sqrt(area) / (width * 0.1)  # Normalize by image width
    depth = base_depth / max(size_factor, 0.1)  # Avoid division by zero
    
    return max(depth, 0.5)  # Minimum depth of 0.5 meters

def pixel_to_3d(pixel_x, pixel_y, depth, frame_shape):
    """
    Convert 2D pixel coordinates to 3D world coordinates
    """
    height, width = frame_shape[:2]
    
    # Set principal point to image center if not already set
    global cx, cy
    if cx == 0:
        cx = width / 2
    if cy == 0:
        cy = height / 2
    
    # Convert pixel coordinates to normalized camera coordinates
    x_norm = (pixel_x - cx) / fx
    y_norm = (pixel_y - cy) / fy
    
    # Convert to 3D world coordinates
    X = x_norm * depth
    Y = y_norm * depth
    Z = depth
    
    return X, Y, Z

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
    # Centroids 
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def get_contour_3d_info(contour, frame):
    """
    Get 3D information for a contour including center and depth
    """
    center_2d = get_contour_center(contour)
    depth = estimate_depth_monocular(frame, contour)
    center_3d = pixel_to_3d(center_2d[0], center_2d[1], depth, frame.shape)
    
    return {
        'center_2d': center_2d,
        'center_3d': center_3d,
        'depth': depth,
        'area': cv2.contourArea(contour),
        'contour': contour
    }

def process_contours(_contours, frame):
    filtered_contours = []
    objects_3d = []
    
    for _contour in _contours:
        if filter_contour(_contour):
            filtered_contours.append(_contour)
            
            # Get 3D information
            obj_3d = get_contour_3d_info(_contour, frame)
            objects_3d.append(obj_3d)
            
            center = obj_3d['center_2d']
            depth = obj_3d['depth']
            center_3d = obj_3d['center_3d']
            
            # Center marker
            above_center = (center[0] - 20, center[1] - 20)
            cv2.putText(img=frame, text="Center", org=above_center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0,0,0), thickness=1)
            cv2.drawMarker(img=frame, position=center, color=(0,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
            
            # Label 2D coords and depth below center
            below_center = (center[0] - 50, center[1] + 30)
            coord_text = f"2D: {center}"
            cv2.putText(img=frame, text=coord_text, org=below_center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=1)
            
            # Add 3D coordinates and depth
            below_center_3d = (center[0] - 50, center[1] + 70)
            coord_3d_text = f"3D: ({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f})"
            cv2.putText(img=frame, text=coord_3d_text, org=below_center_3d, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,255), thickness=1)
            
            # Add depth information
            below_center_depth = (center[0] - 50, center[1] + 100)
            depth_text = f"Depth: {depth:.2f}m"
            cv2.putText(img=frame, text=depth_text, org=below_center_depth, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,0,255), thickness=1)
    
    # Store 3D positions for tracking
    if objects_3d:
        object_3d_positions.append(objects_3d)
    
    return filtered_contours, objects_3d

def draw_contours(frame, contours):
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    return frame

class Plot3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (depth, meters)')
        self.ax.set_title('3D Object Tracking')
        
        # Set axis limits
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 15])
        
        self.points = []
        self.colors = []
        
    def update_plot(self, objects_3d):
        self.ax.clear()
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (depth, meters)')
        self.ax.set_title('3D Object Tracking')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 15])
        
        # Plot current objects
        if objects_3d:
            for obj in objects_3d:
                x, y, z = obj['center_3d']
                self.ax.scatter(x, y, z, c='red', s=100, alpha=0.8)
                self.ax.text(x, y, z, f'  D:{obj["depth"]:.1f}m', fontsize=8)
        
        # Plot trajectory (last few positions)
        if len(object_3d_positions) > 1:
            for i, frame_objects in enumerate(list(object_3d_positions)[-10:]):  # Last 10 frames
                alpha = 0.3 + 0.7 * (i / 10)  # Fade older positions
                for obj in frame_objects:
                    x, y, z = obj['center_3d']
                    self.ax.scatter(x, y, z, c='blue', s=20, alpha=alpha)
        
        plt.draw()
        plt.pause(0.01)

def create_depth_heatmap(frame, objects_3d):
    """
    Create a depth heatmap overlay
    """
    height, width = frame.shape[:2]
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    for obj in objects_3d:
        center = obj['center_2d']
        depth = obj['depth']
        # Create a circular region around each object with its depth
        cv2.circle(depth_map, center, 30, depth, -1)
    
    # Normalize and convert to color map
    if np.max(depth_map) > 0:
        depth_map_norm = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
        
        # Blend with original frame
        mask = depth_map > 0
        result = frame.copy()
        result[mask] = cv2.addWeighted(frame[mask], 0.7, depth_colored[mask], 0.3, 0)
        return result
    
    return frame

# Initialize 3D plot
plt.ion()  # Turn on interactive mode
plot_3d = Plot3D()

showing_full_frame = False
show_3d_plot = False
show_depth_heatmap = False

print("Controls:")
print("'s' - Toggle between full frame and filtered view")
print("'3' - Toggle 3D plot window")
print("'d' - Toggle depth heatmap overlay")
print("'q' - Quit")
paused = False
while True:
    ret, frame = video.read()
    if not ret:
        break
        
    grey_scale = apply_filters(frame)
    contours = find_contours(grey_scale)
    filtered_contours, objects_3d = process_contours(contours, frame)
    
    # Handle keyboard input
    # Python doesn't have a do-while loop, but to execute the following block at least once,
    # and then repeat while paused is True, use:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            showing_full_frame = not showing_full_frame
            print(f"Switched to {'full frame' if showing_full_frame else 'filtered'} view")
        elif key == ord('3'):
            show_3d_plot = not show_3d_plot
            print(f"3D plot {'enabled' if show_3d_plot else 'disabled'}")
            if not show_3d_plot:
                plt.close('all')
                plot_3d = Plot3D()
        elif key == ord('d'):
            show_depth_heatmap = not show_depth_heatmap
            print(f"Depth heatmap {'enabled' if show_depth_heatmap else 'disabled'}")
        elif key == ord('q'):
            break
        elif key == ord('p'):
            print(f"Objects 3D: {objects_3d}")
            paused = not paused
        if not paused:
            break
    # Prepare display frame
    if showing_full_frame:
        display_frame = frame.copy()
        draw_contours(display_frame, filtered_contours)
        if show_depth_heatmap and objects_3d:
            display_frame = create_depth_heatmap(display_frame, objects_3d)
    else:
        display_frame = grey_scale.copy()
        if len(display_frame.shape) == 2:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        draw_contours(display_frame, filtered_contours)
    
    # Add 3D info overlay
    if objects_3d:
        info_text = f"Objects detected: {len(objects_3d)}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show depth range
        depths = [obj['depth'] for obj in objects_3d]
        depth_range = f"Depth range: {min(depths):.1f}m - {max(depths):.1f}m"
        cv2.putText(display_frame, depth_range, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.imshow("3D Object Tracking", display_frame)
    
    # # Update 3D plot
    # if show_3d_plot and objects_3d:
    #     try:
    #         plot_3d.update_plot(objects_3d)
    #     except Exception as e:
    #         print(f"3D plot error: {e}")
    #         show_3d_plot = False

print("Cleaning up...")
video.release()
cv2.destroyAllWindows()
plt.close('all')