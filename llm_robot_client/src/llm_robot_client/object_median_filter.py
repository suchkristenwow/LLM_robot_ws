#!/usr/bin/env python2

import numpy as np
from scipy import stats
import Queue as queue 
from math import sqrt
import weightedstats as ws
import copy
import json

# ROS Imports
import rospy

# from marble_object_detection_msgs.msg import Object, ObjectArray, PotentialObject
from llm_robot_client.msg import ProjectedDetectionArray, ProjectedDetection
from visualization_msgs.msg import Marker, MarkerArray


class ObjectMedianFilter:
    def __init__(self):
        # Publish rate
        self.rate = rospy.Rate(rospy.get_param("~median_filter_rate", 20))
        self.robot = rospy.get_param("~vehicle", default="H01")

        # Publishers
        filtered_objects_topic = rospy.get_param(
            "~all_objects_topic", "object_detector/projected_objects"
        )
        self.object_array_publisher = rospy.Publisher(
            filtered_objects_topic, ProjectedDetectionArray, latch=True, queue_size=10
        )
        self.object_array_visualizer_publisher = rospy.Publisher(
            filtered_objects_topic + "/visualizer",
            MarkerArray,
            latch=True,
            queue_size=10,
        )

        # Subscribers
        potential_object_topic = rospy.get_param(
            "~current_objects_topic", "localized_objects"
        )
        self.current_objects_sub = rospy.Subscriber(
            potential_object_topic, ProjectedDetectionArray, self.current_objects_cb
        )

        # Data Storage
        self.potential_object_queue = queue.Queue(maxsize=1000)

        # Thresholds (Comes from dict in YAML param file)
        # Reads thresholds from files
        # Load file names from rosparameters
        # Object Threshold
        # object_thresholds_file = rospy.get_param('~object_thresholds_file','object_thresholds.txt')
        # Distance Threshold
        # distance_thresholds_file = rospy.get_param('~distance_thresholds_file','distance_thresholds.txt')
        # Num Views Threshold
        # num_views_thresholds_file = rospy.get_param('~num_view_thresholds_file','view_thresholds.txt')
        # Convert files into python dictionaries using JSON
        # self.object_thresholds = self.create_dict_from_file(object_thresholds_file)
        self.distance_threshold = rospy.get_param("~distance_threshold", default=0.1)
        self.num_views = 2

        # Create storage arrays
        self.all_object_array = ProjectedDetectionArray()
        self.all_object_array_view_filtered = ProjectedDetectionArray()
        self.object_array_filter = ProjectedDetectionArray()
        # self.object_array_filtered = ProjectedDetectionArray()
        self.object_visualizer = MarkerArray()
        rospy.loginfo("Object Median Filter Initialized")

        # Create a map from object filtered to all_object_array
        self.map = []

    def create_dict_from_file(self, filename):
        with open(filename) as f:
            data = f.read()
        return json.loads(data)

    def run(self):
        self.process_potential_object()
        self.all_object_array_view_filtered.header.stamp = rospy.Time.now()
        self.all_object_array_view_filtered.header.frame_id = "world"
        self.object_array_publisher.publish(self.all_object_array_view_filtered)
        self.object_array_visualizer_publisher.publish(self.object_visualizer)

    """Callback for potential objects (maybe only visual?)"""

    def current_objects_cb(self, current_object_array):
        rospy.logdebug("Recieved Current Artifcat List")
        for object in current_object_array.detections:
            rospy.logdebug("Adding object to queue %s", object.name)
            self.potential_object_queue.put(object)

    """ Check if an object is within an existing distance based on threshold
    of another object"""

    def check_distances_and_type(self, new_object):
        rospy.logdebug("Checking distances")
        # Set distance from current object to other objects to infinity
        dist = float("inf")
        # Say object matches no other objects
        idx_return = None
        # elif (new_object.name is not "phone") and (new_object.name is not "gas") and (new_object.name is not "cube"):
        # Check the object array
        for idx, object in enumerate(self.object_array_filter.detections):
            # Check all positions in the object array
            rospy.logdebug(
                "Checking object %s to new_object name %s",
                self.object_array_filter.detections[idx].name,
                new_object.name,
            )
            if self.object_array_filter.detections[idx].name == new_object.name:
                for idx2, pos in enumerate(object.position.x):
                    # Comput the distance from the new object to a sample of an object in the array
                    x_diff = object.position.x[idx2] - new_object.position.x
                    y_diff = object.position.y[idx2] - new_object.position.y
                    z_diff = object.position.z[idx2] - new_object.position.z
                    current_dist = sqrt(x_diff**2 + y_diff**2 + z_diff**2)
                    rospy.logdebug("Current Distance %s", current_dist)
                    # If that distance is less than the threshold distance for the given class
                    if current_dist < (self.distance_threshold) and current_dist < dist:
                        rospy.logdebug("Setting distance")
                        # Set that as the current distance
                        dist = current_dist
                        # Set the index of the object
                        idx_return = idx

        return idx_return

    def add_object(self, idx, current_object):
        rospy.logdebug("Adding object %s position to list", current_object.name)
        self.object_array_filter.detections[idx].position.x.append(
            current_object.position.x
        )
        self.object_array_filter.detections[idx].position.y.append(
            current_object.position.y
        )
        self.object_array_filter.detections[idx].position.z.append(
            current_object.position.z
        )
        self.object_array_filter.detections[idx].confidence.append(
            current_object.confidence
        )

    def object_filter(self, idx):
        # Run a median filter on the updated object
        rospy.logdebug("Filtering Object")
        self.all_object_array.detections[idx].position.x = copy.deepcopy(
            float(
                ws.weighted_median(
                    self.object_array_filter.detections[idx].position.x,
                    weights=self.object_array_filter.detections[idx].confidence,
                )
            )
        )
        self.all_object_array.detections[idx].position.y = copy.deepcopy(
            float(
                ws.weighted_median(
                    self.object_array_filter.detections[idx].position.y,
                    weights=self.object_array_filter.detections[idx].confidence,
                )
            )
        )
        self.all_object_array.detections[idx].position.z = copy.deepcopy(
            float(
                ws.weighted_median(
                    self.object_array_filter.detections[idx].position.z,
                    weights=self.object_array_filter.detections[idx].confidence,
                )
            )
        )
        self.all_object_array.detections[idx].confidence = copy.deepcopy(
            float(ws.mean(self.object_array_filter.detections[idx].confidence))
        )
 
        # Check if we have seen the objects the same number of times given by the threshold
        if len(self.object_array_filter.detections[idx].position.x) == self.num_views:
            temp_object = copy.deepcopy(self.all_object_array.detections[idx])
            self.all_object_array_view_filtered.detections.append(
                copy.deepcopy(temp_object)
            )
            # Track where in the array the object has been added
            num_objects = len(self.all_object_array_view_filtered.detections) - 1
            rospy.logdebug(
                "Adding tuple to mapping id:%s num_art:%s",
                str(idx),
                str(num_objects),
            )
            temp_tuple = (idx, num_objects)
            self.map.append(temp_tuple)

            # Visualizer Object via marker
            current_object_marker = Marker()
            current_object_marker.header.frame_id = "world"
            current_object_marker.type = current_object_marker.CYLINDER
            current_object_marker.action = current_object_marker.ADD
            current_object_marker.id = num_objects
            current_object_marker.scale.x = 0.2
            current_object_marker.scale.y = 0.2
            current_object_marker.scale.z = 0.2
            current_object_marker.color.a = 1.0
            current_object_marker.color.r = 1.0
            current_object_marker.color.g = 1.0
            current_object_marker.color.b = 0.0
            current_object_marker.pose.orientation.w = 1.0
            current_object_marker.pose.position.x = (
                self.all_object_array_view_filtered.detections[num_objects].position.x
            )
            current_object_marker.pose.position.y = (
                self.all_object_array_view_filtered.detections[num_objects].position.y
            )
            current_object_marker.pose.position.z = (
                self.all_object_array_view_filtered.detections[num_objects].position.z
            )
            self.object_visualizer.markers.append(current_object_marker)

        """Commented out because we don't actually want positions to change after we have have set them (I do not think this is a good idea)"""
        # # Check if we have seen the object more than the number of times given by the threshold
        # if len(self.object_array_filter.detections[idx].position.x)>self.num_views[self.all_object_array.detections[idx].name]:
        #     # if so update the existing object based on the map
        #     for elem in self.map:
        #         if elem[0] == idx:
        #             object_num = elem[1]
        #             self.all_object_array_view_filtered.detections[object_num] = copy.deepcopy(self.all_object_array.detections[idx])
        #             self.object_visualizer.markers[object_num].pose.position.x = self.all_object_array_view_filtered.detections[object_num].position.x
        #             self.object_visualizer.markers[object_num].pose.position.y = self.all_object_array_view_filtered.detections[object_num].position.y
        #             self.object_visualizer.markers[object_num].pose.position.z = self.all_object_array_view_filtered.detections[object_num].position.z

    """Processes all objects from the queue. Calls all the steps to add them to the final
    object array"""

    def process_potential_object(self):
        # 1 Pop Potential Object From queue
        while not self.potential_object_queue.empty():
            try:
                current_object = self.potential_object_queue.get(timeout=1)
                rospy.logdebug("Object Class %s", current_object.name)
                if current_object is not None:
                    # 1 Check if object already in array
                    rospy.logdebug("Checking if object already in array")
                    # gets index of object if it is in the array
                    idx = self.check_distances_and_type(current_object)

                    # If we find an index add measurement to the filter and update
                    if idx is not None:
                        rospy.logdebug("IDXFOUND")
                        self.add_object(idx, current_object)
                        self.object_filter(idx)

                    else:
                        # num_objects = len(self.object_array_filter.detections)
                        # current_time = str(rospy.get_rostime().secs)[-4:]
                        # current_object.object_id = (
                        #     self.robot
                        #     + "_"
                        #     + str(num_objects)
                        #     + "_"
                        #     + current_time
                        # )
                        self.all_object_array.detections.append(
                            copy.deepcopy(current_object)
                        )
                        rospy.logdebug("Adding object to list")
                        rospy.logdebug("Object Class %s", current_object.name)
                        rospy.logdebug(
                            "Current All Object Array %s", self.all_object_array.detections
                        )
                        # self.all_object_array.num_objects = (
                        #     self.all_object_array.num_objects + 1
                        # )
                        current_object.position.x = [current_object.position.x]
                        current_object.position.y = [current_object.position.y]
                        current_object.position.z = [current_object.position.z]
                        current_object.confidence = [current_object.confidence]
                        # current_object.name = [current_object.name]
                        self.object_array_filter.detections.append(current_object)
                        # self.object_array_filter.num_objects = (
                        #     self.object_array_filter.num_objects + 1
                        # )
            except queue.Empty:
                pass
