import rospy
from llm_robot_client.srv import getPixelFrom3D,get3DPointFromPixel, ProjectBackOfBbox
from geometry_msgs.msg import Point

def get_3d_point_from_pixel(u, v, cam_frame):
    # Wait for the service to become available
    rospy.wait_for_service('/get_3d_point_from_pixel')
    
    try:
        # Create a service proxy
        get_3d_point_service = rospy.ServiceProxy('/get_3d_point_from_pixel', Get3DPointFromPixel)
        
        # Call the service with pixel coordinates and camera frame
        response = get_3d_point_service(u, v, cam_frame)
        
        # Handle the response
        if response:
            world_point = response.world_point
            rospy.loginfo("3D World Point: x=%f, y=%f, z=%f", world_point.x, world_point.y, world_point.z)
            return world_point
        else:
            rospy.logwarn("Service call failed or no valid 3D point was returned.")
            return None
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", str(e))
        return None

if __name__ == "__main__":
    rospy.init_node('get_3d_point_from_pixel_client')

    px_coord = (403,525) 
    u = 403; v = 525
    cam_frame = "cam_front_link"  # Replace with the actual camera frame (e.g., "left", "right", "front")

    # Call the function to get the 3D point
    world_point = get_3d_point_from_pixel(u, v, cam_frame)

    if world_point:
        print("Received 3D world coordinates: x={}, y={}, z={}".format(world_point.x, world_point.y, world_point.z))
    else:
        print("Failed to get 3D world coordinates.")
