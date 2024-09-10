/*
This node localizes Detections of an object in camera space by projecting from camera space to xyz space
*/

#include <llm_robot_client/project_detections.h>

int main(int argc, char** argv){
    if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)){
        ros::console::notifyLoggerLevelsChanged();
        }
    ROS_INFO("[Project Detections] Initializing Node.");
    ros::init(argc, argv, "project_Detections");
    ros::NodeHandle nh;
    const ros::NodeHandle nh_private("~");
    ROS_DEBUG("[Project Detections]: Node Handle nampspace: %s", nh.getNamespace().c_str());
    
    tf2_ros::Buffer* tf_buffer = new tf2_ros::Buffer(ros::Duration(360.0));
    tf2_ros::TransformListener tf_listener(*tf_buffer);

    ros::Rate loop_rate(15);
    ProjectDetections localizer(nh, nh_private, tf_buffer, &tf_listener) ;
    localizer.initialize_params();


    while(ros::ok()){
        localizer.run();
        //ROS_DEBUG("Spinning");
        ros::spinOnce();
        loop_rate.sleep();
    }
    ROS_INFO("[Project Detections] Done.");


}
