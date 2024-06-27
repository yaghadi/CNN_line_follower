import rospy
import cv2
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
import numpy as np

# Import your model definition and ImageFilter
from model_train_kernel import LineFollowerCNN, ImageFilter

# Initialize ROS node, publisher, and bridge
rospy.init_node('line_follower_node')
bridge = CvBridge()
command_pub = rospy.Publisher("motor_commands", String, queue_size=10)

# Load the trained model
model = LineFollowerCNN()
model.load_state_dict(torch.load('line_follower_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ImageFilter(filter_type='sobel')
])

def predict(image):
    # Convert OpenCV image to PIL Image
    pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply transforms
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    
    class_names = ['GO', 'LEFT', 'RIGHT']
    return class_names[predicted.item()]

def plan(prediction):
    if prediction == "LEFT":
        command = "LEFT"
    elif prediction == "RIGHT":
        command = "RIGHT"
    elif prediction == "GO":
        command = "GO"
    else:
        command = "STOP"
    
    print(f"Prediction: {prediction}, Command: {command}")
    
    # Publish command
    command_pub.publish(command)

def imgCallback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        # Make prediction
        prediction = predict(cv_image)
        
        # Plan based on prediction
        plan(prediction)
        
        # Display image (optional)
        cv2.imshow("Raw Image", cv_image)
        cv2.waitKey(3)
    
    except CvBridgeError as e:
        print(e)

def main():
    print("Line Follower Node Started!")
    img_sub = rospy.Subscriber("/camera/image_raw", Image, imgCallback)
    rospy.spin()

if __name__ == "__main__":
    main()
