#!/usr/bin/env python2
import rospy
from llm_robot_client.object_detection_yolo_client import ObjectDetectionClient as YOLOObjectDetectionClient


if __name__ == '__main__':
    rospy.init_node('object_detection_client_node')
    rate = rospy.Rate(20)  # Set the desired frequency (30Hz)

    detector_name = rospy.get_param('detector_name', "yolo")
    od_client = YOLOObjectDetectionClient()

    def shutdown_hook():
        rospy.loginfo("Shutting down Object Detection Client node...")

    rospy.on_shutdown(shutdown_hook)
    
    #rospy.spin()
    while not rospy.is_shutdown():
        # Perform object detection tasks here
        rate.sleep()
