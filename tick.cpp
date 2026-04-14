#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "geometry_msgs/msg/twist.hpp"

class TickController : public rclcpp::Node
{
public:
    TickController()
    : Node("tick_controller"), left_ticks_(0), right_ticks_(0), stopped_(false)
    {
        left_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "left_ticks", 10,
            std::bind(&TickController::leftCallback, this, std::placeholders::_1));

        right_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "right_ticks", 10,
            std::bind(&TickController::rightCallback, this, std::placeholders::_1));

        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&TickController::controlLoop, this));
    }

private:
    void leftCallback(const std_msgs::msg::Int32::SharedPtr msg)
    {
        left_ticks_ = msg->data;
    }

    void rightCallback(const std_msgs::msg::Int32::SharedPtr msg)
    {
        right_ticks_ = msg->data;
    }

    void controlLoop()
    {
        geometry_msgs::msg::Twist cmd;

        int avg_ticks = (left_ticks_ + right_ticks_) / 2;

        if (avg_ticks >= 1800)
        {
            if (!stopped_) {
                RCLCPP_INFO(this->get_logger(), "Reached 1800 ticks. Stopping.");
                stopped_ = true;
            }
            cmd.linear.x = 0.0;
        }
        else
        {
            cmd.linear.x = 0.1;
        }

        cmd.angular.z = 0.0;
        cmd_pub_->publish(cmd);
    }

    // Subscribers
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr left_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr right_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    int left_ticks_;
    int right_ticks_;
    bool stopped_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TickController>());
    rclcpp::shutdown();
    return 0;
}