#include <ros/ros.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

void pose_callback(const geometry_msgs::PoseStampedConstPtr &pose)
{

  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = pose->header.stamp;
  transformStamped.header.frame_id = "semantic_segmentation_world";
  transformStamped.child_frame_id = "semantic_rgbd_camera/link/rgbd_camera";
  transformStamped.transform.translation.x = pose->pose.position.x;
  transformStamped.transform.translation.y = pose->pose.position.y;
  transformStamped.transform.translation.z = pose->pose.position.z;

  // rotate 90 degrees around x-axis
  tf2::Quaternion q;
  q.setRPY(-M_PI / 2, 0, -M_PI / 2);
  // q.setRPY(0, 0, 0);

  // Take the pose angle and add 90 degrees to it
  tf2::Quaternion q2(pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z, pose->pose.orientation.w);

  // IMPORTANT NOTE START
  // Try commenting lines 33 and 34. You will see that the gazebo point cloud now aligns with the world frame but the camera is not aligned with the point cloud
  // MAKE SURE TO catkin_make after changing the code !!!
  q2 = q2 * q;    // rotate the quaternion
  q2.normalize(); // normalize the quaternion
  // IMPORTANT NOTE END

  transformStamped.transform.rotation.x = q2.x();
  transformStamped.transform.rotation.y = q2.y();
  transformStamped.transform.rotation.z = q2.z();
  transformStamped.transform.rotation.w = q2.w();

  br.sendTransform(transformStamped);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "camera_pose_transformer");

  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe("/model/semantic_rgbd_camera/pose", 10, &pose_callback);
  ros::Publisher pub = nh.advertise<geometry_msgs::PoseStamped>("/model/semantic_rgbd_camera/pose", 10);

  ros::spin();
  return 0;
};