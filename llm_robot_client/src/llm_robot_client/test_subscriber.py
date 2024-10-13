import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def ground_plane_callback(msg):
    rospy.loginfo("Received PointCloud2 message")
    # You can try processing points here
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    rospy.loginfo("Received {} points".format(len(points)))

if __name__ == '__main__':
    rospy.init_node('test_pointcloud_subscriber')
    rospy.Subscriber("/H03/ground_plane_points", PointCloud2, ground_plane_callback)
    rospy.spin()
