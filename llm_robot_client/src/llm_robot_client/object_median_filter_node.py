#!/usr/bin/env python2

import rospy
from llm_robot_client.object_median_filter import ObjectMedianFilter

def shutdown_hook():
    rospy.loginfo("Shutting down Artifact Fusion node...")

if __name__ == "__main__":
    rospy.init_node("artifact_fusion", log_level=rospy.INFO)
    rospy.loginfo("Starting the run")
    rate = rospy.Rate(rospy.get_param("~median_filter_rate", 20))
    artifact_fusion = ObjectMedianFilter() 
    rospy.on_shutdown(shutdown_hook) 
    while not rospy.is_shutdown():
        artifact_fusion.run()
        rate.sleep()


