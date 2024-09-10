#!/usr/bin/env python2

import rospy
import requests
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import Header, String, Bool
from llm_robot_client.msg import ObjectDetection, ObjectDetectionArray
import copy
from flask import Flask, request, jsonify
import cv2
import numpy as np
import threading
from collections import deque
import message_filters
import re   

"""
class CircularBuffer(deque):
    def __init__(self, size=0):
        super().__init__(maxlen=size)

    def put(self, item):
        self.append(item)
"""

class CircularBuffer(deque):
    def __init__(self, size=0):
        # In Python 2, you have to use the old-style super() or call the parent class directly
        deque.__init__(self, maxlen=size)

    def put(self, item):
        self.append(item)

class ObjectDetectionClient:
    def __init__(self):
        # Initialize ROS Node
        cam_front_topic = rospy.get_param("~cam_front_topic", "image")
        cam_left_topic = rospy.get_param("~cam_left_topic", "image")
        cam_right_topic = rospy.get_param("~cam_right_topic", "image")
        rospy.loginfo("cam_front_topic: %s", cam_front_topic)
        rospy.loginfo("cam_left_topic: %s", cam_left_topic)
        rospy.loginfo("cam_right_topic: %s", cam_right_topic)
        object_list_topic = rospy.get_param("~object_list_topic", "object_list")
        self.crop_threshold = rospy.get_param("~crop_threshold", 15)
        self.publish_concate_image_bool = rospy.get_param(
            "~publish_concate_image", True
        )
        self.publish_detections_image_bool = rospy.get_param(
            "~publish_detections_image", True
        )
        self.object_detection_server_url = rospy.get_param(
            "~object_detection_server_url", "http://0.0.0.0.edu:5000"
        )
        
        self.object_detector_status_topic = rospy.get_param(
            "~object_detector_status_topic", "object_detector/server_status"
        )

        self.image_front_sub = message_filters.Subscriber(
            cam_front_topic, CompressedImage
        )
        self.image_left_sub = message_filters.Subscriber(
            cam_left_topic, CompressedImage
        )
        self.image_right_sub = message_filters.Subscriber(
            cam_right_topic, CompressedImage
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_front_sub, self.image_left_sub, self.image_right_sub],
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self.image_callback)
        self.image_queue = CircularBuffer(size=2)

        # Object List Callback
        """
        self.object_list_subscriber = rospy.Subscriber(
            object_list_topic, String, self.receive_object_labels
        )
        """
        
        self.concate_image_publisher = rospy.Publisher(
            "object_detector/concate/image/compressed",
            CompressedImage,
            latch=True,
            queue_size=10,
        )
        
        object_detection_topic = rospy.get_param("~object_detection_2D_topic", "object_detection")

        self.detected_image_publisher = rospy.Publisher(
            "object_detector/detection/image/compressed",
            CompressedImage,
            latch=True,
            queue_size=10,
        )
        
        rospy.loginfo("object_detection_topic: %s", object_detection_topic)

        self.detections_publisher = rospy.Publisher(
            object_detection_topic,
            ObjectDetectionArray,
            queue_size=10,
            latch=True,
        )
        
        object_detector_status_topic = rospy.get_param("~object_detector_status_topic", "object_detector/server_status")
        
        self.server_status_publisher = rospy.Publisher(
            object_detector_status_topic,
            Bool,
            queue_size=10,
            latch=True,
        )

        # CV Bridge
        self.bridge = CvBridge()

        self.object_list = ["fire extinguisher"]
        
        self.cam_frame_ids = {'right': None, 'front': None, 'left': None}
        
        self.frame_width = None
        
        self.init_detection_thread()

        rospy.loginfo("[object_detection_client] loaded and ready")
        
        self.server_running = False

    def init_detection_thread(self):
        self.detection_lock = threading.Lock()
        self.detection_thread = threading.Thread(target=self.run_detection_thread)
        self.detection_thread.start()

    def run_detection_thread(self):
        while not rospy.is_shutdown():
            # Acquire the lock
            acquired = self.detection_lock.acquire(False)  # Non-blocking acquisition
            
            try:
                if acquired:  # Proceed only if the lock was acquired
                    if len(self.image_queue) > 0 and self.object_list is not None:
                        image_data = copy.deepcopy(self.image_queue.pop())
                        self.detect_objects(image_data)
                    else:
                        rospy.logwarn("Cannot detect objects ...")
                        if self.object_list is None:
                            rospy.loginfo("Object List is None!")
                        if len(self.image_queue) == 0:
                            rospy.loginfo("Image queue is empty")
                else:
                    rospy.logwarn("Failed to acquire lock")
                    
            except Exception as e:
                rospy.logerr("Error in detection thread: {}".format(e))
            
            finally:
                # Ensure the lock is only released if it was acquired
                if acquired and self.detection_lock.locked():
                    self.detection_lock.release()
            
            rospy.sleep(0.05)
            
    def image_callback(self, front, left, right):
        """
        Callback function for processing image messages.
        Called from time synchronizer.

        Args:
            msg: The image message received.
            front: The front image message.
            left: The left image message.
            right: The right image message.
        """
        cv_image_front = self.bridge.compressed_imgmsg_to_cv2(
            front, desired_encoding="bgr8"
        )
        cv_image_left = self.bridge.compressed_imgmsg_to_cv2(
            left, desired_encoding="bgr8"
        )
        cv_image_right = self.bridge.compressed_imgmsg_to_cv2(
            right, desired_encoding="bgr8"
        )
        
        images = {
            'front': cv_image_front,
            'left': cv_image_left,
            'right': cv_image_right
        }
        
        # Check if frame_ids have been gathered
        if self.cam_frame_ids['front'] is None:
            self.cam_frame_ids['front'] = front.header.frame_id
        if self.cam_frame_ids['left'] is None:
            self.cam_frame_ids['left'] = left.header.frame_id
        if self.cam_frame_ids['right'] is None:
            self.cam_frame_ids['right'] = right.header.frame_id
        
        self.image_queue.put({'time': front.header.stamp, 'image': images})

    def concatenate_images(self, front, left, right):
        """
        Concatenates three images horizontally and publishes the resulting image.

        Args:
            front (numpy.ndarray): The front image.
            left (numpy.ndarray): The left image.
            right (numpy.ndarray): The right image.

        Returns:
            numpy.ndarray: The concatenated image.
        """
        threshold = 15
        left = copy.deepcopy(left)
        right = copy.deepcopy(right)
        front = copy.deepcopy(front)
        left_image = self.crop_image(left, threshold)
        front_image = self.crop_image(front, threshold)
        right_image = self.crop_image(right, threshold)
        single_image = np.concatenate((left_image, front_image, right_image), axis=1)
        if self.publish_concate_image:
            self.publish_concate_image(single_image)
        return single_image

    def crop_image(self, image, threshold):
        """
        Crop the input image by setting the left and right columns to zero (black).

        Args:
            image (numpy.ndarray): The input image to be cropped.
            threshold (int): The number of columns to be set to zero on each side.

        Returns:
            numpy.ndarray: The cropped image.
        """
        image[:, :threshold] = 0  # Left
        image[:, -threshold:] = 0  # Right
        return image

    def receive_object_labels(self, msg):
        #rospy.loginfo("Object list received")
        self.object_list = msg.data
        
    def encode_image(self, image):
        """
        Encodes the input image to JPEG format.

        Args:
            image (numpy.ndarray): The input image to be encoded.

        Returns:
            bytes: The encoded image.
        """
        success, encoded_image = cv2.imencode(".jpg", image)
        if not success:
            rospy.loginfo("Error encoding image")
            return None
        return encoded_image
    

    def detect_objects(self, image_data):
        image_list = ['front', 'left', 'right']
        rospy.loginfo("detecting objects...")
        for frame in image_list:
            try:
                encoded_image = self.encode_image(image_data['image'][frame])
                cv2.imwrite("/home/marble/LLM_robot_ws/src/llm_robot_client/frame.jpg",image_data['image'][frame])
                # Prepare the headers for the HTTP request
                headers = {"Content-Type": "image/jpeg"}
                rospy.loginfo("Posting request to yolo world server ...") 
                # Send the image to the GLIP server
                response = requests.post(
                    self.object_detection_server_url + "/process",
                    data=encoded_image.tobytes(),
                    headers=headers,
                    params={"caption": self.object_list},
                )
                print("response: ",response) 
                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the response
                    #rospy.loginfo("GLIP response received: {}".format(response))
                    self.process_detection_results(image_data, response.json(), frame)
                else:
                    rospy.loginfo(
                        "Error in GLIP server response: {}".format(response.status_code)
                    )
            except requests.exceptions.RequestException as e:
                rospy.loginfo("Error querying GLIP server: {}".format(e))
            

    def process_detection_results(self, image_data, response_data, frame):
        # Publish the server is runnig if not already done
        if self.server_running == False:
            self.server_status_publisher.publish(True)
            self.server_running = True
        
        # Extracting data from the response
        x_coords = response_data["x_coords"]
        y_coords = response_data["y_coords"]
        scores = response_data["scores"]
        labels = response_data["labels"]
        caption = response_data["caption"]
        rospy.logdebug("x_coords: {}".format(x_coords))
        rospy.logdebug("y_coords: {}".format(y_coords))
        rospy.logdebug("scores: {}".format(scores))
        rospy.logdebug("labels: {}".format(labels))
        rospy.logdebug("caption: {}".format(caption))
        
        # Creating an ObjectDetectionArray message
        detection_array = ObjectDetectionArray()
        detection_array.header = Header()  # You might want to fill in more details here
        detection_array.header.stamp = image_data['time']

        # Iterating through detections and adding them to the array
        for i in range(len(labels)):
            detection = ObjectDetection()
            detection.header = (
                detection_array.header
            )  # Using the same header as the array
            detection.name = labels[i]
            detection.confidence = scores[i]
            detection.x = int((x_coords[i][0] + x_coords[i][1]) / 2)
            detection.y = int((y_coords[i][0] + y_coords[i][1]) / 2)
            detection.left = int(x_coords[i][0])
            detection.lower = int(y_coords[i][1])
            detection.right = int(x_coords[i][1])
            detection.upper = int(y_coords[i][0])
            
            # Get frame
            detection.header.frame_id = self.cam_frame_ids[frame]
            if detection.header.frame_id is None:
                rospy.logerr('Frame ID is None') 
                continue

            # Adding the detection to the array
            detection_array.detections.append(detection)

        self.detections_publisher.publish(detection_array)
        if self.publish_detections_image_bool and detection_array.detections:
            self.publish_detections_image(image_data['image'][frame], detection_array)
            
    def get_frame_id(self, x):
        if  x < self.frame_width:
            rospy.logdebug('left')
            return 'left'  # Frame ID of the left camera
        elif x < self.frame_width * 2:
            rospy.logdebug('front')
            return 'front'  # Frame ID of the front camera
        else:
            rospy.logdebug('right')
            return 'right'
        
    def get_center_x(self, x):
        
        if x < self.frame_width:
            return x   # Coordinate is in the left image
        elif x < self.frame_width * 2:
            return x - self.frame_width  # Coordinate is in the front image
        else:
            return x - self.frame_width*2  # Coordinate is in the right image

    def publish_concate_image(self, image):
        """
        Publishes the concatenated image.

        Args:
            image (numpy.ndarray): The concatenated image.
        """
        image_msg = self.bridge.cv2_to_compressed_imgmsg(image)
        self.concate_image_publisher.publish(image_msg)

    def publish_detections_image(self, image, detections):
        """
        Draws bounding boxes and labels on the image.

        Args:
        - image (numpy.ndarray): The image on which to draw, in OpenCV format.
        - detections (list of ObjectDetection): The list of detections. Each detection
        should have attributes 'left', 'lower', 'right', 'upper', 'name'.

        Returns:
        - An image with detections drawn.
        """
        for detection in detections.detections:
            # Coordinates of the bounding box
            x_min = detection.left
            y_min = detection.upper
            x_max = detection.right
            y_max = detection.lower

            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Prepare the label text
            #label_text = f"{detection.name}: {detection.confidence:.2f}"
            label_text = str(detection.name) + ": " + str(np.round(detection.confidence,2)) 

            # Compute text size to create a background for it
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image,
                (x_min, y_min - text_height - 5),
                (x_min + text_width, y_min),
                (0, 255, 0),
                -1,
            )

            # Put the label text on the image
            cv2.putText(
                image,
                label_text,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        self.detected_image_publisher.publish(
            self.bridge.cv2_to_compressed_imgmsg(image)
        )


