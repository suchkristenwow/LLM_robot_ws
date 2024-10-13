/**
 * Projects the center of a bounding box into free space
 **/

#include <llm_robot_client/project_detections.h>

/**
 * Ros Callbacks
 * */

void ProjectDetections::initialize_params()
{
    std::vector<std::string> keys;
    _nh_private.getParamNames(keys);
    // Generate map of keys to values
    for (std::string s : keys)
    {
        std::string next_param;
        _nh_private.getParam(s, next_param);
        _param_vals.insert({s, getParamAsString(s)});
    }
    ROS_INFO("[Project Detections] Read Parameters");
    // Print out values for testing
    // TODO prevent iteration of this loop if lower than debug level
    for (auto const &s : _param_vals)
    {
        // ROS_DEBUG("[Project Detections] Read Param: %s with a value of %s ", s.first.c_str(), s.second.c_str());
    }
}

std::string ProjectDetections::getParamAsString(auto val)
{
    std::string next_param;
    if (_nh_private.getParam(val, next_param))
    {
        return next_param;
    }
    float next_param_f;
    if (_nh_private.getParam(val, next_param_f))
    {
        return std::to_string(next_param_f);
    }
    double next_param_d;
    if (_nh_private.getParam(val, next_param_d))
    {
        return std::to_string(next_param_d);
    }
    int next_param_i;
    if (_nh_private.getParam(val, next_param_i))
    {
        return std::to_string(next_param_i);
    }
    bool next_param_b;
    if (_nh_private.getParam(val, next_param_b))
    {
        return std::to_string(next_param_b);
    }
    return "false";
}

/**
 *Initialize Subscribers
 **/
void ProjectDetections::initialize_subscribers()
{
    _pc_sub = _nh.subscribe("horiz/os_cloud_node/points",1, &ProjectDetections::point_cloud_callback, this); 

    // Center Point Sub
    std::string object_detection_topic;
    _nh_private.param<std::string>("object_detection_2D_topic", object_detection_topic, "object_detections");
    _object_detections_sub = _nh.subscribe(object_detection_topic, 10, &ProjectDetections::object_detections_callback, this);

    // Octomap
    std::string octomap_topic;
    _nh_private.param<std::string>("octomap_topic", octomap_topic, "marble_mapping");
    _octomap_sub = _nh.subscribe(octomap_topic, 10, &ProjectDetections::map_callback, this);

    // Get the number of cameras
    int num_cam;
    _nh_private.param("num_cam", num_cam, 1);

    // ROS_WARN("[Project Detections] Initializing subscribers ... This is the number of cameras: %d", num_cam); 

    for (int i = 0; i < num_cam; i++)
    {
        std::string cam_info_topic;
        // ROS_WARN("[Project Detections] Trying to subscribe to: %s", "cam" + std::to_string(i) + "_info");
        if (_nh_private.getParam("cam" + std::to_string(i) + "_info", cam_info_topic))
        {
            // ROS_WARN("[Project Detections] Subscribing to %s topic", cam_info_topic.c_str());
            ros::Subscriber current_sub = _nh.subscribe(cam_info_topic, 10, &ProjectDetections::cam_info_callback, this);
            _cam_info_sub_array.push_back(current_sub);
        }
        else
        {
            ROS_ERROR("[Project Detections]: Failed to get param %s", ("cam" + std::to_string(i) + "_info").c_str());
        }
    }
}

/**
 * Initialize Publishers
 * */

void ProjectDetections::initialize_publishers()
{
    ground_plane_publisher = _nh.advertise<sensor_msgs::PointCloud2>("ground_plane_points", 1); 
    ground_plane_z_publisher = _nh.advertise<std_msgs::Float32>("ground_level_z",1);
    marker_pub = _nh.advertise<visualization_msgs::Marker>("projected_detection_marker", 1);
    _ray_publisher = _nh.advertise<visualization_msgs::Marker>("camera_projected_ray", 1);
    _transform_publisher = _nh.advertise<geometry_msgs::TransformStamped>("camera_transfrom", 1);
    _ray_transform_publisher = _nh.advertise<geometry_msgs::PoseStamped>("camera_pose_ray", 1);
    std::string current_objects_topic;
    _nh_private.param<std::string>("current_objects_topic", current_objects_topic, "object_detector/current_detections");
    _projected_detection_publisher = _nh.advertise<llm_robot_client::ProjectedDetectionArray>(current_objects_topic, 1);
    _rivz_pub = _nh.advertise<visualization_msgs::MarkerArray>("object_detector/projection_viz", 1);
    _debug_rviz_pub = _nh.advertise<visualization_msgs::Marker>("object_detector/debug_rviz", 1);
}

/**
 * Octomap Callback
 * Takes in an octomap and stores it in the object.
 * Topic is controlled in launch file
 */

void ProjectDetections::point_cloud_callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    _cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *_cloud);
}

void ProjectDetections::map_callback(const octomap_msgs::Octomap::ConstPtr &msg)
{
    // ROS_DEBUG("[Localize Artfiacts:] Recieved Map");
    if (!_have_map)
    {
        _have_map = true;
    }
    _map = *msg;

    // Convert the binary message to an octomap::AbstractOcTree pointer
    octomap::AbstractOcTree* tree = octomap_msgs::binaryMsgToMap(*msg);
    if (tree)
    {
        // Cast to RoughOcTree if necessary (based on your use case)
        _tree = dynamic_cast<octomap::RoughOcTree*>(tree); 
        if (!_tree)
        {
            ROS_ERROR("Failed to cast OctoMap to RoughOcTree");
            delete tree;
        }
    }
    else
    {
        ROS_ERROR("Failed to convert octomap message to AbstractOcTree");
    }
}

/**
 * Object Detections Callback
 */

void ProjectDetections::object_detections_callback(const llm_robot_client::ObjectDetectionArray::ConstPtr &msg)
{
    ROS_DEBUG("[Project Coordiantes:] Recieved Object Detection Array");
    _detection_queue.push(*msg); 
}

/**
 * Camera Info Callback
 * Takes in the camera info messages from each camera stream
 * Model is stored in a map containg the frame id and the model
 */

void ProjectDetections::cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr &msg)
{
    ROS_INFO("[Project Detections] Entered cam info callback ..."); 
    std::string cam_frame = msg->header.frame_id;
    ROS_INFO("Received CameraInfo for frame: %s", cam_frame.c_str());
    // If Cam frame not in map add it
    if (!_cam_models_map.count(cam_frame))
    {
        // ROS_WARN("[Project Detections]: Insesrting %s into camera map.", cam_frame.c_str());
        image_geometry::PinholeCameraModel current_camera_model;
        current_camera_model.fromCameraInfo(*msg);
        _cam_models_map.emplace(cam_frame, current_camera_model);
    } else {
        ROS_INFO("Camera frame %s already exists in the map.", cam_frame.c_str());
    }
}

/**
 * Projects the camera into a ray in xy coordiantes
 * Note this is in the camera frame X(right), Y(down), Z(forward)
 */

cv::Point3d ProjectDetections::project_cam_xy(std::string &cam_frame, cv::Point2d &center_point)
{
    cv::Point3d projected_ray(-1, -1, -1);
    ROS_DEBUG("Project Function cam model: %s", cam_frame.c_str());
    if (_cam_models_map.count(cam_frame))
    {
        ROS_DEBUG("Found camera");
        auto current_model = _cam_models_map.at(cam_frame);
        cv::Point2d rectified_point;
        bool rectify_point;
        _nh_private.param<bool>("rectify_point", rectify_point, false);
        if (rectify_point)
        {
            rectified_point = current_model.rectifyPoint(center_point);
        }
        projected_ray = current_model.projectPixelTo3dRay(center_point);
    }

    return projected_ray;
}

bool ProjectDetections::lookup_transform(std::string &frame1, std::string &frame2, ros::Time time, geometry_msgs::TransformStamped &transfrom)
{
    // ROS_WARN("looking up requested transform between %s and %s ... ",frame1.c_str(),frame2.c_str()); 
    try
    {
        transfrom = _tf_buffer->lookupTransform(frame1, frame2, time, ros::Duration(1.0));
        return true;
    }
    catch (tf2::TransformException &ex)
    {
        try {
            transfrom = _tf_buffer->lookupTransform(frame1, frame2, ros::Time(0), ros::Duration(1.0));
            return true; 
        } catch (tf2::TransformException &ex) {
            ROS_DEBUG("TF Lookup failed %s", ex.what());
            ROS_WARN("%s", ex.what());
            return false;
        }
    }
}

// Apply transform to camera ray
bool ProjectDetections::project_cam_world(cv::Point3d initial_ray, std::string &reference_frame, std::string &cam_frame, ros::Time time, cv::Point3d &world_ray)
{

    geometry_msgs::TransformStamped transform;
    geometry_msgs::TransformStamped transform_back;

    auto current_namespace = _nh.getNamespace(); 

    if (!current_namespace.empty() && current_namespace[0] == '/') {
            current_namespace.erase(0, 1);  // Remove the leading '/'
        } 

    if (!(cam_frame.find(current_namespace) != std::string::npos)){
        cam_frame = current_namespace + "/" + std::string(cam_frame);
    }

    // ROS_WARN("Calling lookup transform with world_frame_id: %s and cam_frame_id: %s", reference_frame.c_str(), cam_frame.c_str()); 
    if (lookup_transform(reference_frame, cam_frame, time, transform))
    {
        ROS_DEBUG("Projecting Cam Ray into World Ray");
        ROS_DEBUG("Transfrom X: %f, Y:%f, Z: %f", transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);
        geometry_msgs::Vector3 initial_ray_vector;
        initial_ray_vector.x = initial_ray.z;
        initial_ray_vector.y = -initial_ray.x;
        initial_ray_vector.z = -initial_ray.y;

        geometry_msgs::PoseStamped initial_ray_pose;
        geometry_msgs::Quaternion initial_ray_orientation;
        initial_ray_orientation.w = 1;
        initial_ray_pose.pose.position.x = initial_ray.z;
        initial_ray_pose.pose.position.y = -initial_ray.x;
        initial_ray_pose.pose.position.z = -initial_ray.y;
        initial_ray_pose.pose.orientation = initial_ray_orientation;
        geometry_msgs::PoseStamped world_ray_pose;

        // Vector
        geometry_msgs::Vector3 world_ray_vector;
        ROS_DEBUG("World Ray before transform X: %f, Y: %f Z: %f", world_ray_vector.x, world_ray_vector.y, world_ray_vector.z);
        tf2::doTransform(initial_ray_vector, world_ray_vector, transform);

        ROS_DEBUG("World Ray after transform X: %f, Y: %f Z: %f", world_ray_vector.x, world_ray_vector.y, world_ray_vector.z);
        //_ray_transform_publisher.publish(world_ray_pose);
        _transform_publisher.publish(transform);
        // world_ray.x =  world_ray_pose.pose.position.x;
        // world_ray.y = world_ray_pose.pose.position.y;
        // world_ray.z = world_ray_pose.pose.position.z;
        world_ray.x = world_ray_vector.x;
        world_ray.y = world_ray_vector.y;
        world_ray.z = world_ray_vector.z;

        return true;
    } 
    return false;
}

bool ProjectDetections::create_projected_detection(octomap::point3d &point, llm_robot_client::ObjectDetection &detection, llm_robot_client::ProjectedDetection &projected_detection)
{
    projected_detection.header.frame_id = "world";
    projected_detection.position.x = point.x();
    projected_detection.position.y = point.y();
    projected_detection.position.z = point.z();
    projected_detection.name = detection.name;
    projected_detection.image = detection.image;
    projected_detection.confidence = detection.confidence;

    // Make sure contour points are already set in the projected_detection outside this function
    // Publish a marker to visualize the projected center point
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "projected_detection";
    marker.id = 0;  // Unique identifier for this marker
    marker.type = visualization_msgs::Marker::SPHERE;  // You can also use Marker::CUBE, etc.
    marker.action = visualization_msgs::Marker::ADD;

    // Set marker pose (position and orientation)
    marker.pose.position.x = point.x();
    marker.pose.position.y = point.y();
    marker.pose.position.z = point.z();
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // Set marker scale (adjust size as needed)
    marker.scale.x = 0.1;  // Diameter of the sphere
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    // Set marker color (adjust color and alpha as needed)
    marker.color.r = 1.0;  // Red
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;  // Alpha (opacity)

    // Publish the marker
    marker_pub.publish(marker);

    return true;
}

/*
 * Used to publish a point to rviz for visualization
 */

void ProjectDetections::publish_point(octomap::point3d &point)
{
    visualization_msgs::Marker direction_viz;
    direction_viz.header.frame_id = "world";
    direction_viz.header.stamp = ros::Time();
    direction_viz.type = visualization_msgs::Marker::SPHERE;
    direction_viz.action = visualization_msgs::Marker::ADD;
    direction_viz.pose.position.x = point.x();
    direction_viz.pose.position.y = point.y();
    direction_viz.pose.position.z = point.z();
    direction_viz.pose.orientation.x = 0.0;
    direction_viz.pose.orientation.y = 0.0;
    direction_viz.pose.orientation.z = 0.0;
    direction_viz.pose.orientation.w = 1.0;
    direction_viz.scale.x = 0.5;
    direction_viz.scale.y = 0.5;
    direction_viz.scale.z = 0.5;
    direction_viz.color.a = 1.0; // Don't forget to set the alpha!
    direction_viz.color.r = 1.0;
    direction_viz.color.g = 0.0;
    direction_viz.color.b = 0.0;
    direction_viz.lifetime = ros::Duration(100);
    ROS_DEBUG("[Project Detections]: Publishing ray");
    _ray_publisher.publish(direction_viz);
}

void ProjectDetections::publish_ground_plane_cloud(){
    // Apply a PassThrough filter to remove points with z-values greater than 0.2 meters
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-std::numeric_limits<float>::max(), 0.0); // Keep points with z <= 0.1
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*filtered_cloud);

    // Segment the largest plane in the filtered point cloud
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(filtered_cloud); // Use the filtered cloud
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        ROS_WARN("Could not estimate a planar model for the given dataset.");
    } else {
        // Extract the Z component (assuming the plane is horizontal)
        float a = coefficients->values[0];
        float b = coefficients->values[1];
        float c = coefficients->values[2]; // Z component coefficient
        float d = coefficients->values[3];

        // Calculate the Z intercept (assuming plane equation: ax + by + cz + d = 0)
        // Z = -(a*x + b*y + d)/c, when x = 0, y = 0
        float z_value = -d / c;

        // Publish the ground plane Z value
        std_msgs::Float32 z_msg;
        z_msg.data = z_value;
        ground_plane_z_publisher.publish(z_msg);
        ROS_INFO("Published ground plane Z value: %f", z_value);
    }

    // Extract the plane points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(filtered_cloud);
    extract.setIndices(inliers);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*ground_plane_cloud);

    // Convert the extracted ground plane points to a ROS PointCloud2 message
    sensor_msgs::PointCloud2 ground_plane_msg;
    pcl::toROSMsg(*ground_plane_cloud, ground_plane_msg);
    ground_plane_msg.header.frame_id = "/H03/base_link";  // Replace with your frame ID
    ground_plane_msg.header.stamp = ros::Time::now();

    // Publish the ground plane points
    ground_plane_publisher.publish(ground_plane_msg);
    ROS_INFO("Published ground plane points.");
} 

void ProjectDetections::publish_projected_detections(llm_robot_client::ProjectedDetectionArray &projected_detections)
{
    _projected_detection_publisher.publish(projected_detections);
}

/**
 * Rounds a point to the nearest point in an octomap.
 * Points are rounded based on the resolution of the octomap
 */

void ProjectDetections::round_point(octomap::point3d &point, double resolution)
{
    resolution = 100.0 * resolution;
    point.x() = std::round(point.x() * resolution) / resolution;
    point.y() = std::round(point.y() * resolution) / resolution;
    point.z() = std::round(point.z() * resolution) / resolution;
}

void ProjectDetections::publish_detection_arrray_to_rviz(const llm_robot_client::ProjectedDetectionArray &detections)
{
    visualization_msgs::MarkerArray marker_array;
    int id = 0;

    for (const auto &detection : detections.detections)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "base_link"; // Set the frame ID to your robot's frame
        marker.header.stamp = ros::Time::now();
        marker.ns = "projected_detections";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::SPHERE; // Choose the type of marker (e.g., SPHERE, CUBE, ARROW)
        marker.action = visualization_msgs::Marker::ADD;

        // Set the position of the marker
        marker.pose.position.x = detection.position.x;
        marker.pose.position.y = detection.position.y;
        marker.pose.position.z = detection.position.z;

        // Set the orientation of the marker
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        // Set the scale of the marker
        marker.scale.x = 0.2; // Change these values as needed
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;

        // Set the color of the marker
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        marker.lifetime = ros::Duration(); // Duration(0) means the marker never auto-deletes

        marker_array.markers.push_back(marker);
    }

    _rivz_pub.publish(marker_array);
}

void ProjectDetections::debug_rviz(octomap::point3d &origin, octomap::point3d &direction)
{
    // Create markers
    visualization_msgs::Marker origin_marker, direction_marker;

    // Set the frame ID and timestamp
    origin_marker.header.frame_id = direction_marker.header.frame_id = "world";
    origin_marker.header.stamp = direction_marker.header.stamp = ros::Time::now();

    // Set the namespace and id for the markers
    origin_marker.ns = "origin";
    direction_marker.ns = "direction";
    origin_marker.id = 0;
    direction_marker.id = 1;

    // Set the marker type to SPHERE and ARROW respectively
    origin_marker.type = visualization_msgs::Marker::SPHERE;
    direction_marker.type = visualization_msgs::Marker::ARROW;

    // Set the scale of the markers
    origin_marker.scale.x = 0.2;
    origin_marker.scale.y = 0.2;
    origin_marker.scale.z = 0.2;
    direction_marker.scale.x = 0.5; // shaft diameter
    direction_marker.scale.y = 0.2; // head diameter
    direction_marker.scale.z = 0.3; // head length

    // Set the color
    origin_marker.color.r = 1.0;
    origin_marker.color.g = 0.0;
    origin_marker.color.b = 0.0;
    origin_marker.color.a = 1.0;

    direction_marker.color.r = 0.0;
    direction_marker.color.g = 1.0;
    direction_marker.color.b = 0.0;
    direction_marker.color.a = 1.0;

    // Set the pose of the markers
    // For origin
    origin_marker.pose.position.x = origin.x();
    origin_marker.pose.position.y = origin.y();
    origin_marker.pose.position.z = origin.z();
    origin_marker.pose.orientation.w = 1.0;

    // For direction
    direction_marker.pose.position = origin_marker.pose.position;
    tf::Quaternion q;
    q.setRPY(0, 0, atan2(direction.y(), direction.x()));
    tf::quaternionTFToMsg(q, direction_marker.pose.orientation);

    // Publish the markers
    _debug_rviz_pub.publish(origin_marker);
    _debug_rviz_pub.publish(direction_marker);
}

bool ProjectDetections::project_detection(llm_robot_client::ObjectDetection &detection, octomap::RoughOcTree *current_octree, llm_robot_client::ProjectedDetection &projected_detection)
{
    auto current_namespace = _nh.getNamespace(); 
    if (!current_namespace.empty() && current_namespace[0] == '/') {
            current_namespace.erase(0, 1);  // Remove the leading '/'
        } 

    std::string cam_frame = detection.header.frame_id;
    std::string cam_frame_id = cam_frame;

    // List to store projected contour points
    std::vector<geometry_msgs::Point> projected_contour_points;

    // Project center point of the bounding box
    cv::Point2d center_point(detection.x, detection.y);
    ROS_DEBUG("[Project Detections]: Processing center point with %d, %d, Frame %s",
              detection.x, detection.y, cam_frame_id.c_str());

    auto projected_cam_ray = project_cam_xy(cam_frame, center_point);
    ROS_DEBUG("[Project Detections]: Projected center ray from camera in LOCAL: %s, %s, %s.", 
              std::to_string(projected_cam_ray.x).c_str(), std::to_string(projected_cam_ray.y).c_str(), std::to_string(projected_cam_ray.z).c_str());

    std::string reference_frame = "world";
    bool cam_ray_status = project_cam_world(projected_cam_ray, reference_frame, cam_frame_id, ros::Time::now(), projected_cam_ray);
    
    if (cam_ray_status)
    {
        std::string lidar_link = _nh_private.param("lidar_link", lidar_link);
        geometry_msgs::TransformStamped lidar_to_world;
        std::string lidar_frame_id = current_namespace + "/" + lidar_link;
        std::string world_frame_id = "world";

        // ROS_WARN("Calling lookup transform with world_frame_id: %s and lidar_frame_id: %s", world_frame_id.c_str(), lidar_frame_id.c_str());
        if (lookup_transform(world_frame_id, lidar_frame_id, ros::Time::now(), lidar_to_world))
        {
            octomap::point3d artifact_location;
            octomap::point3d origin(lidar_to_world.transform.translation.x, lidar_to_world.transform.translation.y, lidar_to_world.transform.translation.z);
            octomap::point3d direction(projected_cam_ray.x, projected_cam_ray.y, projected_cam_ray.z);

            if (current_octree != NULL)
            {
                double projection_distance = 12.0;
                bool projection_status = current_octree->castRay(origin, direction, artifact_location, true, projection_distance);

                // If projection fails, find intersection with ground plane for center point
                if (!projection_status)
                {
                    octomap::point3d intersection;
                    bool intersection_status = findIntersectionWithGround(origin, direction, intersection); 
                    //bool intersection_status = findIntersectionWithGround(origin, direction, artifact_location); 
                    if (intersection_status)
                    {
                        ROS_INFO("Found intersection with the ground plane for center point!");
                        create_projected_detection(intersection, detection, projected_detection);
                    }
                    else
                    {
                        ROS_DEBUG("[Project Detections]: Failed to project ray or find ground plane for center point");
                    }
                }
                else
                {
                    create_projected_detection(artifact_location, detection, projected_detection);
                }

                // Iterate over contour points and project them
                for (const auto &contour_point : detection.contour_pts)
                {
                    cv::Point2d contour_px(contour_point.x, contour_point.y);

                    auto contour_ray = project_cam_xy(cam_frame, contour_px);
                    bool contour_ray_status = project_cam_world(contour_ray, reference_frame, cam_frame_id, ros::Time::now(), contour_ray);

                    if (contour_ray_status)
                    {
                        octomap::point3d contour_location;
                        octomap::point3d contour_direction(contour_ray.x, contour_ray.y, contour_ray.z);

                        bool contour_projection_status = current_octree->castRay(origin, contour_direction, contour_location, true, projection_distance);

                        // If projection fails for contour point, try ground plane intersection
                        if (!contour_projection_status)
                        {
                            octomap::point3d contour_intersection;
                            bool contour_intersection_status = findIntersectionWithGround(origin, contour_direction, contour_intersection); 

                            if (contour_intersection_status)
                            {
                                ROS_INFO("Found intersection with the ground plane for contour point!");
                                geometry_msgs::Point projected_point;
                                projected_point.x = contour_intersection.x();
                                projected_point.y = contour_intersection.y();
                                projected_point.z = contour_intersection.z();
                                projected_contour_points.push_back(projected_point);
                            }
                            else
                            {
                                ROS_DEBUG("[Project Detections]: Failed to project ray or find ground plane for contour point");
                            }
                        }
                        else
                        {
                            // Successfully projected contour point
                            geometry_msgs::Point projected_point;
                            projected_point.x = contour_location.x();
                            projected_point.y = contour_location.y();
                            projected_point.z = contour_location.z();
                            projected_contour_points.push_back(projected_point);
                        }
                    }
                }

                // Add the projected contour points to the detection
                projected_detection.contour_pts = projected_contour_points;

                return true;
            }
        }
    }
    return false;
}

/**
 * Main Procesing fucntion
 * 1) Takes the latest detection from the queue of bouding boxes
 * 2) Projects to XY space
 * 3) Performs raycast
 * 4) Publishes artifact
 */
void ProjectDetections::run()
{
    publish_ground_plane_cloud(); 
    // If there are detections in the queue process them
    if (!_detection_queue.empty())
    {
        // Obtain current detection
        auto current_detection_array = _detection_queue.front();
        // Remove from queue
        _detection_queue.pop();
        // Get the octomap
        auto current_map = octomap_msgs::binaryMsgToMap(_map);
        auto current_octomap = dynamic_cast<octomap::RoughOcTree *>(current_map);
        llm_robot_client::ProjectedDetectionArray projected_detections;
        for (auto current_detection : current_detection_array.detections)
        {
            llm_robot_client::ProjectedDetection projected_detection;
            if (project_detection(current_detection, current_octomap, projected_detection))
            {
                projected_detections.detections.push_back(projected_detection);
            }
        }
        if (projected_detections.detections.size() > 0)
        {
            publish_detection_arrray_to_rviz(projected_detections);
        }
        publish_projected_detections(projected_detections);

        // delete octomap
        delete current_octomap;
        // Publish Results
    }
}

bool ProjectDetections::findIntersectionWithGround(const octomap::point3d &origin, const octomap::point3d &direction, octomap::point3d &intersection) {
    // Apply a PassThrough filter to remove points with z-values greater than 0.2 meters
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-std::numeric_limits<float>::max(), 0.0); // Keep points with z <= 0.1
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*filtered_cloud);

    // Segment the largest plane in the filtered point cloud
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(filtered_cloud); // Use the filtered cloud
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        ROS_WARN("Could not estimate a planar model for the given dataset.");
        return false;
    }

    // Output the z-value of the plane's normal vector
    float z_normal = coefficients->values[2];
    float d = coefficients->values[3];
    //ROS_INFO("Ground plane normal z-value: %f, plane coefficient d: %f", z_normal, d);

    // Convert direction to Eigen vector
    Eigen::Vector3f ray_direction(direction.x(), direction.y(), direction.z());
    Eigen::Vector3f ray_origin(origin.x(), origin.y(), origin.z());

    // Compute the intersection of the ray with the plane
    float t = -(coefficients->values[0] * ray_origin.x() + coefficients->values[1] * ray_origin.y() + coefficients->values[2] * ray_origin.z() + d) / 
              (coefficients->values[0] * ray_direction.x() + coefficients->values[1] * ray_direction.y() + coefficients->values[2] * ray_direction.z());

    if (t >= 0) {
        Eigen::Vector3f intersection_point = ray_origin + t * ray_direction;
        intersection.x() = intersection_point.x();
        intersection.y() = intersection_point.y();
        intersection.z() = intersection_point.z();
        return true;
    }

    ROS_WARN("No intersection found with the ground plane.");
    return false;
} 

bool ProjectDetections::get_pixel_from_3d(
    llm_robot_client::GetPixelFrom3D::Request &req,
    llm_robot_client::GetPixelFrom3D::Response &res)
{
    // Get the camera frame from the request
    //std::string cam_frame = req.cam_frame;
    std::string cam_frame = "cam_front_link"; 

    // Check if the camera frame is present in the camera models map
    if (_cam_models_map.count(cam_frame) == 0)
    {
        for (const auto &cam : _cam_models_map)
        {
            ROS_WARN("Available camera frame: %s", cam.first.c_str());
        }

        ROS_ERROR("Camera frame %s not found!", cam_frame.c_str());
        return false;
    }
    
    // Get the camera model
    auto current_model = _cam_models_map.at(cam_frame);

    // ROS_WARN("Camera intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
        //  current_model.fx(), current_model.fy(),
        //  current_model.cx(), current_model.cy());
 
    // Lookup the transform from world frame to camera frame
    geometry_msgs::TransformStamped transform;

    auto current_namespace = _nh.getNamespace(); 

    if (!current_namespace.empty() && current_namespace[0] == '/') {
            current_namespace.erase(0, 1);  // Remove the leading '/'
        } 

    if (!(cam_frame.find(current_namespace) != std::string::npos)){
        cam_frame = current_namespace + "/" + std::string(cam_frame);
    }
    
    try {
        transform = _tf_buffer->lookupTransform("world", cam_frame, ros::Time(0), ros::Duration(1.0));
        // ROS_WARN("Transform from world to %s: [%.2f, %.2f, %.2f] rotation: [%.2f, %.2f, %.2f, %.2f]",
        //      cam_frame.c_str(),
        //      transform.transform.translation.x,
        //      transform.transform.translation.y,
        //      transform.transform.translation.z,
        //      transform.transform.rotation.x,
        //      transform.transform.rotation.y,
        //      transform.transform.rotation.z,
        //      transform.transform.rotation.w); 
    } catch (tf2::TransformException &ex) {
        ROS_WARN("Transform failed: %s", ex.what());
        return false;
    }

    // ROS_WARN("This is requested 3D point: (%.2f,%.2f,%.2f)",req.x,req.y,req.z); 
    // Transform the 3D point from world frame to camera frame
    tf2::Vector3 world_point(req.x, req.y, req.z); 

    tf2::Vector3 cam_point = tf2::Transform(tf2::Quaternion(
        transform.transform.rotation.x, transform.transform.rotation.y,
        transform.transform.rotation.z, transform.transform.rotation.w),
        tf2::Vector3(
        transform.transform.translation.x, transform.transform.translation.y,
        transform.transform.translation.z)).inverse() * world_point;

    // Project the transformed 3D point (in the camera frame) to 2D pixel coordinates
    cv::Point3d transformed_point(-cam_point.y(), -cam_point.z(), cam_point.x());

    //ROS_WARN("Transformed 3D point in camera frame: (%.2f, %.2f, %.2f)", cam_point.x(), cam_point.y(), cam_point.z());
 
    cv::Point2d pixel = current_model.project3dToPixel(transformed_point);
    
    // Set the pixel coordinates in the response
    res.u = pixel.x;
    res.v = pixel.y;

    return true;
}
    
void ProjectDetections::initialize_services()
{
    // Define the service
    _get_world_to_pixel_service = _nh_private.advertiseService(
        "get_pixel_from_3d", &ProjectDetections::get_pixel_from_3d, this);
    ROS_INFO("[project detections] Service get_pixel_from_3d initialized!");
    // Initialize the service that projects pixels to world points
    _get_pixel_to_world_service = _nh_private.advertiseService(
        "get_3d_point_from_pixel", &ProjectDetections::get_3d_point_from_pixel, this);
    ROS_INFO("[project detections] Service get3DPointFromPixel initialized!");
    // Initialize the service that projects pixels to world points
    _get_back_of_bbox_service = _nh_private.advertiseService(
        "get_back_of_bbox_from_pixel", &ProjectDetections::get_back_of_bbox_from_pixel, this);
    ROS_INFO("[project detections] Service get_back_of_bbox_from_pixel initialized!");
}

bool ProjectDetections::get_3d_point_from_pixel(
    llm_robot_client::Get3DPointFromPixel::Request &req,
    llm_robot_client::Get3DPointFromPixel::Response &res)
{
    std::string cam_frame = req.cam_frame;
    std::string cam_frame_id = req.cam_frame;

    ROS_INFO("Entered the service call!");

    auto current_namespace = _nh.getNamespace(); 
    if (!current_namespace.empty() && current_namespace[0] == '/') {
            current_namespace.erase(0, 1);  // Remove the leading '/'
        } 

    // Check if the camera frame is present in the camera models map
    if (_cam_models_map.count(cam_frame) == 0)
    {

        ROS_ERROR("Camera frame %s not found!", cam_frame.c_str());
        return false;
    }

    // Get the camera model
    auto current_model = _cam_models_map.at(cam_frame);

    // Create a 2D point using the provided pixel coordinates
    cv::Point2d pixel(req.u, req.v);

    // Project the 2D pixel to a 3D ray in the camera frame
    cv::Point3d projected_cam_ray = project_cam_xy(cam_frame, pixel);

    // Now, project the ray from the camera frame into the world frame
    cv::Point3d world_ray;

    std::string reference_frame = "world";
    bool cam_ray_status = project_cam_world(projected_cam_ray, reference_frame, cam_frame_id, ros::Time::now(), projected_cam_ray);

    if (cam_ray_status)
    {
        std::string lidar_link = _nh_private.param("lidar_link", lidar_link);
        geometry_msgs::TransformStamped lidar_to_world;
        std::string lidar_frame_id = current_namespace + "/" + lidar_link;
        std::string world_frame_id = "world";

        // ROS_WARN("Calling lookup transform with world_frame_id: %s and lidar_frame_id: %s", world_frame_id.c_str(), lidar_frame_id.c_str());
        if (lookup_transform(world_frame_id, lidar_frame_id, ros::Time::now(), lidar_to_world))
        {
            octomap::point3d artifact_location;
            octomap::point3d origin(lidar_to_world.transform.translation.x, lidar_to_world.transform.translation.y, lidar_to_world.transform.translation.z);
            octomap::point3d direction(projected_cam_ray.x, projected_cam_ray.y, projected_cam_ray.z);

            if (_tree != NULL)
            {
                double projection_distance = 12.0;
                bool projection_status = _tree->castRay(origin, direction, artifact_location, true, projection_distance);

                // If projection fails, find intersection with ground plane for center point
                if (!projection_status)
                {
                    octomap::point3d intersection;
                    bool intersection_status = findIntersectionWithGround(origin, direction, intersection); 
                    if (intersection_status)
                    {
                        ROS_INFO("Found intersection with the ground plane for center point!");
                        //create_projected_detection(intersection, detection, projected_detection);
                        res.world_point.x = intersection.x();
                        res.world_point.y = intersection.y(); 
                        res.world_point.z = intersection.z(); 
                    }
                    else
                    {
                        ROS_DEBUG("[Project Detections]: Failed to project ray or find ground plane for center point");
                    }
                }
                else
                {
                    //create_projected_detection(artifact_location, detection, projected_detection);
                    res.world_point.x = artifact_location.x();
                    res.world_point.y = artifact_location.y(); 
                    res.world_point.z = artifact_location.z(); 
                }

                return true;
            }
        }
    }
    return false;
}

/*
    const octomap::point3d& front_point, 
    const octomap::point3d& direction, 
    octomap::RoughOcTree* octree, 
    octomap::point3d& back_point);
*/

bool ProjectDetections::find_back_of_bbox(
    const octomap::point3d& front_point, 
    const octomap::point3d& direction, 
    octomap::RoughOcTree* octree, 
    octomap::point3d& back_point) 
{
    // Set a maximum range for the ray
    double max_range = 25.0;  // You can adjust this based on expected object depth
    
    // Perform raycasting
    bool hit = octree->castRay(front_point, direction, back_point, true, max_range);
    
    if (hit) {
        ROS_INFO("Back point found at (%f, %f, %f)", back_point.x(), back_point.y(), back_point.z());
        return true;
    } else {
        ROS_WARN("Raycasting did not find an intersection.");
        return false;
    }
}

bool ProjectDetections::get_back_of_bbox_from_pixel(
    llm_robot_client::ProjectBackOfBbox::Request &req,
    llm_robot_client::ProjectBackOfBbox::Response &res)
{
    if (!_have_map) {
        ROS_ERROR("Map not available.");
        return false;
    }
    ROS_DEBUG("Map is available.");

    std::string cam_frame = req.cam_frame;

    // Project center point of the bounding box
    cv::Point2d center_point(req.u, req.v);

    auto projected_cam_ray = project_cam_xy(cam_frame, center_point);
    ROS_WARN("[Project Detections]: Projected center ray from camera in LOCAL: %s, %s, %s.", 
              std::to_string(projected_cam_ray.x).c_str(), std::to_string(projected_cam_ray.y).c_str(), std::to_string(projected_cam_ray.z).c_str());

    std::string reference_frame = "world";
    bool success = project_cam_world(projected_cam_ray, reference_frame, cam_frame, ros::Time::now(), projected_cam_ray);
    
    if (!success) {
        ROS_ERROR("Failed to project pixel to world space.");
        return false;
    } 

    // Store the front corner in the response
    geometry_msgs::Point front_corner;
    front_corner.x = projected_cam_ray.x;
    front_corner.y = projected_cam_ray.y;
    front_corner.z = projected_cam_ray.z;
    res.front_corners.push_back(front_corner);

    // Find the back of the bounding box using raycasting
    octomap::point3d front_point(projected_cam_ray.x, projected_cam_ray.y, projected_cam_ray.z);
    octomap::point3d direction(0, 0, -1);  // Assuming negative z-direction for the depth
    octomap::point3d back_point;

    // Perform raycasting using Octomap
    if (!_have_map) {
        ROS_ERROR("Map not available.");
        return false;
    }

    //(llm_robot_client::ObjectDetection &detection, octomap::RoughOcTree *current_octree 
    auto current_map = octomap_msgs::binaryMsgToMap(_map);
    auto current_octomap = dynamic_cast<octomap::RoughOcTree *>(current_map);

    bool back_found = find_back_of_bbox(front_point, direction, current_octomap, back_point);

    if (!back_found) {
        ROS_WARN("Failed to find the back of the bounding box.");
        return false;
    }

    // Store the back corner in the response
    geometry_msgs::Point back_corner;
    back_corner.x = back_point.x();
    back_corner.y = back_point.y();
    back_corner.z = back_point.z();
    res.back_corners.push_back(back_corner);

    return true;
}