#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <zed_interfaces/Object.h>
#include <zed_interfaces/ObjectsStamped.h>

std::map<int, geometry_msgs::Point> objectPositions; // Store object positions with label_id as key

void objectListCallback(const zed_interfaces::ObjectsStamped::ConstPtr& msg)
{
  ROS_INFO("***** New object list *****");
  double current_time = ros::Time::now().toSec();

  for (int i = 0; i < msg->objects.size(); i++)
  {
    if (msg->objects[i].label_id == -1)
      continue;

    int label_id = msg->objects[i].label_id;
    geometry_msgs::Point position;
    position.x = msg->objects[i].position[0];
    position.y = msg->objects[i].position[1];
    position.z = msg->objects[i].position[2];

    if (objectPositions.find(label_id) == objectPositions.end()) {
      // If object is not yet in the map, add it
      objectPositions[label_id] = position;
    } else {
      // Calculate velocity based on position difference and time difference
      double dt = current_time - msg->header.stamp.toSec();
      double dx = position.x - objectPositions[label_id].x;
      double dy = position.y - objectPositions[label_id].y;
      double dz = position.z - objectPositions[label_id].z;

      double vx = dx / dt;
      double vy = dy / dt;
      double vz = dz / dt;

      ROS_INFO_STREAM(msg->objects[i].label << " [" << label_id << "] - Velocity: [" << vx << "," << vy << "," << vz << "] m/s");

      // Update object position in the map
      objectPositions[label_id] = position;
    }
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "zed_obj_det_sub_tutorial");
  ros::NodeHandle n;

  ros::Subscriber subObjList = n.subscribe("/zed2i/zed_node/obj_det/objects", 1, objectListCallback);

  ros::spin();

  return 0;
}
