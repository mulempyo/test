#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <cmath>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <message_filters/subscriber.h>

class ICPCompareNode : public rclcpp::Node
{
public:
    ICPCompareNode() : Node("icp_compare_node")
    {
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
         "/scan",
        rclcpp::SensorDataQoS(),
        std::bind(&ICPCompareNode::scanCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered",
            10,
            std::bind(&ICPCompareNode::odomCallback, this, std::placeholders::_1));
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    std::shared_ptr<const sensor_msgs::msg::LaserScan> prev_scan_;
    nav_msgs::msg::Odometry prev_odom_;

    bool has_prev_scan_ = false;
    bool has_prev_odom_ = false;

    Eigen::Matrix4f runICPCUDA(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& current,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& previous,
        int max_iter,
        const Eigen::Matrix4f& init_guess)
    {
        return Eigen::Matrix4f::Identity();
    }

    double normalizeAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        RCLCPP_WARN(this->get_logger(), "in odom callback");
        if (!has_prev_odom_) {
            prev_odom_ = *msg;
            has_prev_odom_ = true;
            return;
        }

        current_odom_ = *msg;
    }

    nav_msgs::msg::Odometry current_odom_;

    void scanCallback(const std::shared_ptr<const sensor_msgs::msg::LaserScan> scan)
    {
        RCLCPP_WARN(this->get_logger(), "in scan callback");
        if (!has_prev_scan_ || !has_prev_odom_) {
            prev_scan_ = scan;
            has_prev_scan_ = true;
            return;
        }

        auto current_scan = laserToCloud(scan);
        auto previous_scan = laserToCloud(prev_scan_);

        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f T = runICPCUDA(current_scan, previous_scan, 20, init_guess);

        double icp_dx = T(0, 3);
        double icp_dy = T(1, 3);
        double icp_theta = std::atan2(T(1,0), T(0,0));

        double x1 = prev_odom_.pose.pose.position.x;
        double y1 = prev_odom_.pose.pose.position.y;

        double x2 = current_odom_.pose.pose.position.x;
        double y2 = current_odom_.pose.pose.position.y;

        double yaw1 = getYaw(prev_odom_);
        double yaw2 = getYaw(current_odom_);

        double odom_dx = x2 - x1;
        double odom_dy = y2 - y1;
        double odom_dtheta = normalizeAngle(yaw2 - yaw1);

        double dx_error = icp_dx - odom_dx;
        double dy_error = icp_dy - odom_dy;
        double dtheta_error = normalizeAngle(icp_theta - odom_dtheta);

        RCLCPP_WARN(this->get_logger(),
            "ICP        : dx=%.4f dy=%.4f dth=%.4f",
            icp_dx, icp_dy, icp_theta);

        RCLCPP_WARN(this->get_logger(),
            "ODOM       : dx=%.4f dy=%.4f dth=%.4f",
            odom_dx, odom_dy, odom_dtheta);

        RCLCPP_WARN(this->get_logger(),
            "ERROR      : dx=%.4f dy=%.4f dth=%.4f",
            dx_error, dy_error, dtheta_error);

        prev_scan_ = scan;
        prev_odom_ = current_odom_;
    }

    double getYaw(const nav_msgs::msg::Odometry& odom)
    {
        const auto& q = odom.pose.pose.orientation;
        return std::atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr laserToCloud(
        const std::shared_ptr<const sensor_msgs::msg::LaserScan> scan)
    {
        auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>());

        double angle = scan->angle_min;

        for (auto r : scan->ranges) {
            if (std::isfinite(r)) {
                pcl::PointXYZ p;
                p.x = r * std::cos(angle);
                p.y = r * std::sin(angle);
                p.z = 0.0;
                cloud->points.push_back(p);
            }
            angle += scan->angle_increment;
        }

        return cloud;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ICPCompareNode>());
    rclcpp::shutdown();
    return 0;
}