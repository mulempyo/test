#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/quaternion.hpp"

using namespace std::chrono_literals;

class SensorMonitorNode : public rclcpp::Node
{
public:
  SensorMonitorNode()
  : Node("sensor_monitor_node")
  {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&SensorMonitorNode::odomCallback, this, std::placeholders::_1));

    filtered_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", 10,
      std::bind(&SensorMonitorNode::filteredCallback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/imu/data", 50,
      std::bind(&SensorMonitorNode::imuCallback, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(
      1000ms, std::bind(&SensorMonitorNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "sensor_monitor_node started.");
  }

private:
  struct Pose2DData
  {
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
    bool valid{false};
  };

  struct ImuData
  {
    double roll{0.0};
    double pitch{0.0};
    double yaw{0.0};
    double wz{0.0};
    bool valid{false};
  };

  // 초기값 관리
  bool odom_has_initial_ = false;
  bool filtered_has_initial_ = false;
  bool imu_has_initial_ = false;

  Pose2DData odom_initial_;
  Pose2DData filtered_initial_;
  ImuData imu_initial_;

  Pose2DData odom_current_;
  Pose2DData filtered_current_;
  ImuData imu_current_;

  static double normalizeAngle(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  static double rad2deg(double rad)
  {
    return rad * 180.0 / M_PI;
  }

  static void quaternionToEuler(
    const geometry_msgs::msg::Quaternion & q,
    double & roll, double & pitch, double & yaw)
  {
    const double x = q.x;
    const double y = q.y;
    const double z = q.z;
    const double w = q.w;

    const double sinr_cosp = 2.0 * (w * x + y * z);
    const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    const double sinp = 2.0 * (w * y - z * x);
    if (std::abs(sinp) >= 1.0)
      pitch = std::copysign(M_PI / 2.0, sinp);
    else
      pitch = std::asin(sinp);

    const double siny_cosp = 2.0 * (w * z + x * y);
    const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
  }

  std::string formatPoseLine(
    const std::string & label,
    const Pose2DData & current,
    bool has_initial,
    const Pose2DData & initial)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    oss << label
        << " | cur: x=" << current.x
        << " y=" << current.y
        << " yaw=" << current.yaw << "rad(" << rad2deg(current.yaw) << "deg)";

    if (has_initial) {
      double dx = current.x - initial.x;
      double dy = current.y - initial.y;
      double dist = std::hypot(dx, dy);
      double dyaw = normalizeAngle(current.yaw - initial.yaw);

      oss << " | delta: dx=" << dx
          << " dy=" << dy
          << " dist=" << dist
          << " dyaw=" << dyaw << "rad(" << rad2deg(dyaw) << "deg)";
    }

    return oss.str();
  }

  std::string formatImuLine(
    const ImuData & current,
    bool has_initial,
    const ImuData & initial)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    oss << "IMU  | cur: roll=" << current.roll
        << " pitch=" << current.pitch
        << " yaw=" << current.yaw << "rad(" << rad2deg(current.yaw) << "deg)"
        << " wz=" << current.wz;

    if (has_initial) {
      double dyaw = normalizeAngle(current.yaw - initial.yaw);

      oss << " | delta_yaw=" << dyaw
          << "rad(" << rad2deg(dyaw) << "deg)";
    }

    return oss.str();
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double roll, pitch, yaw;
    quaternionToEuler(msg->pose.pose.orientation, roll, pitch, yaw);

    odom_current_.x = msg->pose.pose.position.x;
    odom_current_.y = msg->pose.pose.position.y;
    odom_current_.yaw = yaw;
    odom_current_.valid = true;

    if (!odom_has_initial_) {
      odom_initial_ = odom_current_;
      odom_has_initial_ = true;
      RCLCPP_INFO(this->get_logger(), "[ODOM] initial captured");
    }
  }

  void filteredCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double roll, pitch, yaw;
    quaternionToEuler(msg->pose.pose.orientation, roll, pitch, yaw);

    filtered_current_.x = msg->pose.pose.position.x;
    filtered_current_.y = msg->pose.pose.position.y;
    filtered_current_.yaw = yaw;
    filtered_current_.valid = true;

    if (!filtered_has_initial_) {
      filtered_initial_ = filtered_current_;
      filtered_has_initial_ = true;
      RCLCPP_INFO(this->get_logger(), "[EKF] initial captured");
    }
  }

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    double roll, pitch, yaw;
    quaternionToEuler(msg->orientation, roll, pitch, yaw);

    imu_current_.roll = roll;
    imu_current_.pitch = pitch;
    imu_current_.yaw = yaw;
    imu_current_.wz = msg->angular_velocity.z;
    imu_current_.valid = true;

    if (!imu_has_initial_) {
      imu_initial_ = imu_current_;
      imu_has_initial_ = true;
      RCLCPP_INFO(this->get_logger(), "[IMU] initial captured");
    }
  }

  void timerCallback()
  {
    RCLCPP_INFO(this->get_logger(), "----------------------------");

    if (odom_current_.valid)
      RCLCPP_INFO(this->get_logger(), "%s",
        formatPoseLine("ODOM", odom_current_, odom_has_initial_, odom_initial_).c_str());

    if (filtered_current_.valid)
      RCLCPP_INFO(this->get_logger(), "%s",
        formatPoseLine("EKF ", filtered_current_, filtered_has_initial_, filtered_initial_).c_str());

    if (imu_current_.valid)
      RCLCPP_INFO(this->get_logger(), "%s",
        formatImuLine(imu_current_, imu_has_initial_, imu_initial_).c_str());

    if (odom_current_.valid && filtered_current_.valid) {
      double dyaw = normalizeAngle(filtered_current_.yaw - odom_current_.yaw);
      RCLCPP_INFO(this->get_logger(),
        "DIFF EKF-ODOM yaw: %.3f rad (%.2f deg)",
        dyaw, rad2deg(dyaw));
    }
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr filtered_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorMonitorNode>());
  rclcpp::shutdown();
  return 0;
}