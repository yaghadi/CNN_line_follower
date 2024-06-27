import rospy,cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge , CvBridgeError

bridge =CvBridge()
command_pub = rospy.Publisher("motor_commands", String)

def is_yellow(val):
	return val>110

def plan(leftP,rightP):
	command="STOP"
	if is_yellow(leftP) and is_yellow(rightP):
		command="GO"
	if is_yellow(leftP) and not is_yellow(rightP):
		command="LEFT"
	if not is_yellow(leftP) and is_yellow(rightP):
		command="RIGHT"
	if not is_yellow(leftP) and not is_yellow(rightP):
		command="SEARCH"
	print(leftP,rightP,command)


	#publish command
	command_pub.publish(command)
def imgCallback(data):
	cv_image =bridge.imgmsg_to_cv2(data,"bgr8")

	gray_image =cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
	plan(gray_image[700][300],gray_image[700][500])
	gray_image =cv2.line(gray_image,(300,700),(500,700),0,5)
	cv2.imshow("Raw Image",gray_image)
	cv2.waitKey(3)
	
def main():
	print("Hey Universe!")
	rospy.init_node('my_planner_node')
	img_sub=rospy.Subscriber("/camera/image_raw",Image,imgCallback)
	rospy.spin()

if __name__ == "__main__":
   	main()
