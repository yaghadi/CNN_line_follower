import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
command_pub = rospy.Publisher("motor_commands", String, queue_size=10)

# Data collection setup
save_dir = 'data'
os.makedirs(save_dir, exist_ok=True)
for dir in ['LEFT', 'GO', 'RIGHT']:
    os.makedirs(os.path.join(save_dir, dir), exist_ok=True)

image_count = 0
last_yellow_position = None

def is_yellow(val):
    return val > 110

def plan(points, rows, cols):
    global last_yellow_position
    yellow_count = sum(1 for p in points if is_yellow(p))
    total_points = len(points)

    if yellow_count >= total_points * 0.8:  # If 80% or more points are yellow
        return "GO"
    elif yellow_count == 0:
        if last_yellow_position is None:
            return "LEFT"  # Default to LEFT if we've never seen yellow
        elif last_yellow_position < len(cols) // 2:
            return "LEFT"
        else:
            return "RIGHT"
    else:
        left_yellow = sum(1 for r in range(len(rows)) for c in range(len(cols)//2) if is_yellow(points[r*len(cols) + c]))
        right_yellow = sum(1 for r in range(len(rows)) for c in range(len(cols)//2, len(cols)) if is_yellow(points[r*len(cols) + c]))
        
        if left_yellow > right_yellow:
            last_yellow_position = 0  # Left side
            return "LEFT"
        elif right_yellow > left_yellow:
            last_yellow_position = len(cols) - 1  # Right side
            return "RIGHT"
        else:
            return "GO"

def save_image(image, command):
    global image_count
    filename = f"{command}_{image_count}.jpg"
    filepath = os.path.join(save_dir, command, filename)
    cv2.imwrite(filepath, image)
    image_count += 1
    print(f"Saved {filepath}")

def imgCallback(data):
    global image_count, last_yellow_position
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape
        
        # Sample points from multiple rows and columns
        sample_rows = [height - 20, height - 60, height - 100, height - 140, height - 180]  # Look at five different heights
        sample_cols = np.linspace(width//8, 7*width//8, 20, dtype=int)  # 20 sample points across the width
        
        sample_points = []
        for y in sample_rows:
            row_points = [gray_image[y][x] for x in sample_cols]
            sample_points.extend(row_points)
        
        command = plan(sample_points, sample_rows, sample_cols)
        print(f"Command: {command}")
        
        command_pub.publish(command)
        
        # Save the image with its corresponding command
        save_image(cv_image, command)
        
        # Visualization
        for y in sample_rows:
            for x in sample_cols:
                color = (0, 255, 0) if is_yellow(gray_image[y][x]) else (0, 0, 255)
                cv2.circle(cv_image, (x, y), 3, color, -1)
        
        # Draw a vertical line indicating the last known yellow position
        if last_yellow_position is not None:
            cv2.line(cv_image, (sample_cols[last_yellow_position], 0), (sample_cols[last_yellow_position], height), (255, 255, 0), 2)
        
        # Add text showing the current command
        cv2.putText(cv_image, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Processed Image", cv_image)
        cv2.waitKey(3)
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

def main():
    print("Starting line follower and data collection node")
    rospy.init_node('my_planner_node')
    img_sub = rospy.Subscriber("/camera/image_raw", Image, imgCallback)
    rospy.spin()

if __name__ == "__main__":
    main()
