#!/usr/bin/env python2

import rospy
from llm_robot_client.robot import Robot


if __name__ == '__main__':
    rospy.init_node('llm_robot_node')
    rate = rospy.Rate(30)  # Set the desired frequency (30Hz)

    bot = Robot()
    
    def shutdown_hook():
        rospy.loginfo("Shutting down Robot node...")

    rospy.on_shutdown(shutdown_hook)
    
    while not rospy.is_shutdown():
        # Perform object detection tasks here
        bot.run()
        rate.sleep()
