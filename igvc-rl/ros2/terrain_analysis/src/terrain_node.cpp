// SPDX-License-Identifier: BSD-3-Clause
#include "terrain_analysis/cloud_decoder.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <memory>
#include <stdexcept>

namespace terrain_analysis {
class TerrainNode final : public rclcpp::Node {
public:
  TerrainNode() : Node("terrain_analysis") {
    GridConfig c;
    c.cell_size = startup_parameter("cell_size", 0.2);
    c.width = positive_size("grid_width", 105, 501);
    c.max_samples_per_cell = positive_size("max_samples_per_cell", 64, 4096);
    c.min_samples = positive_size("min_samples", 6, 4096);
    c.quantile = startup_parameter("ground_quantile", 0.25);
    c.decay_seconds = startup_parameter("decay_seconds", 2.0);
    c.obstacle_clearance = startup_parameter("obstacle_clearance", 0.15);
    grid_ = std::make_unique<RollingGrid>(c);
    fixed_frame_ = startup_parameter<std::string>("fixed_frame", "map");
    max_scan_age_ = positive_double("max_scan_age", 0.5);
    max_odom_age_ = positive_double("max_odom_age", 0.5);
    max_skew_ = positive_double("max_odom_scan_skew", 0.15);
    future_tolerance_ = positive_double("future_tolerance", 0.05);
    max_points_ = positive_size("max_input_points", 200000, 1000000);
    max_bytes_ = positive_size("max_input_bytes", 32000000, 64000000);
    if (fixed_frame_.empty()) throw std::invalid_argument("fixed_frame cannot be empty");
    rcl_jump_threshold_t jump_threshold{};
    jump_threshold.on_clock_change = false;
    jump_threshold.min_backward.nanoseconds = -1;
    // TimeSource may use its own thread. Defer all grid mutations to an input
    // callback, but remember even resets that happen entirely during a dropout.
    jump_handler_ = get_clock()->create_jump_callback(nullptr,
      [this](const rcl_time_jump_t & jump) {
        if (jump.delta.nanoseconds < 0) clock_reset_pending_.store(true);
      }, jump_threshold);
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/terrain_map", 2);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>("/state_estimation",
      rclcpp::SensorDataQoS().keep_last(10),
      [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { odometry(*m); });
    scan_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>("/registered_scan",
      rclcpp::SensorDataQoS().keep_last(2),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { scan(*m); });
  }

private:
  struct Odom { double stamp, x, y; };
  template<typename T>
  T startup_parameter(const std::string & name, const T & value) {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    return declare_parameter<T>(name, value, descriptor);
  }
  std::size_t positive_size(const char * name, int value, int maximum) {
    const auto v = startup_parameter<std::int64_t>(name, value);
    if (v < 1 || v > maximum) throw std::invalid_argument(name);
    return static_cast<std::size_t>(v);
  }
  double positive_double(const char * name, double value) {
    const auto v = startup_parameter<double>(name, value);
    if (!std::isfinite(v) || v <= 0) throw std::invalid_argument(name);
    return v;
  }
  void reject(const char * why) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Rejected input: %s", why);
  }
  static double seconds(const builtin_interfaces::msg::Time & t) {
    if (t.sec < 0 || t.nanosec >= 1000000000U) return -1;
    return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
  }
  bool timely(double stamp, double now, double age) const {
    return stamp > 0 && now > 0 && now - stamp <= age && stamp - now <= future_tolerance_;
  }
  // ROS time can restart in simulation. Invalidate both spatial evidence and its
  // odometry association before accepting any observation in the new epoch.
  double checked_now() {
    const double current = now().seconds();
    if (clock_reset_pending_.exchange(false) ||
        (last_ros_now_ >= 0 && current + 1e-6 < last_ros_now_)) {
      const auto config = grid_->config();
      grid_ = std::make_unique<RollingGrid>(config);
      odometry_.clear();
      RCLCPP_WARN(get_logger(), "ROS clock reset: terrain and odometry cleared (%.9f -> %.9f)",
        last_ros_now_, current);
    }
    last_ros_now_ = current;
    return current;
  }
  void odometry(const nav_msgs::msg::Odometry & m) {
    const double wall = checked_now();
    const auto t = seconds(m.header.stamp);
    const auto & p = m.pose.pose.position;
    const auto & q = m.pose.pose.orientation;
    const double norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if (m.header.frame_id != fixed_frame_ ||
        !timely(t, wall, max_odom_age_) ||
        !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(norm) || std::abs(norm - 1.0) > 0.05 ||
        std::abs(p.x / grid_->config().cell_size) > 1099511627776.0 ||
        std::abs(p.y / grid_->config().cell_size) > 1099511627776.0 ||
        (!odometry_.empty() && t <= odometry_.back().stamp)) {
      reject("invalid, stale, out-of-order, or wrong-frame odometry"); return;
    }
    odometry_.push_back({t, p.x, p.y});
    if (odometry_.size() > 100) odometry_.pop_front();
  }
  void scan(const sensor_msgs::msg::PointCloud2 & m) {
    const double wall = checked_now(), t = seconds(m.header.stamp);
    if (m.header.frame_id != fixed_frame_ || !timely(t, wall, max_scan_age_) ||
        t <= grid_->current_stamp()) {
      reject("invalid, stale, duplicate, out-of-order, or wrong-frame scan"); return;
    }
    const Odom * best = nullptr;
    for (const auto & odom : odometry_) {
      if (timely(odom.stamp, wall, max_odom_age_) &&
          std::abs(odom.stamp - t) <= max_skew_ &&
          (!best || std::abs(odom.stamp - t) < std::abs(best->stamp - t))) best = &odom;
    }
    if (!best) { reject("no recent, time-matched odometry"); return; }
    CloudLayout layout{m.width, m.height, m.point_step, m.row_step, m.is_bigendian, {}};
    if (m.fields.size() > 128) { reject("too many fields"); return; }
    for (const auto & field : m.fields) {
      if (field.name.size() > 64) { reject("oversized field name"); return; }
      layout.fields.push_back({field.name, field.offset, field.datatype, field.count});
    }
    const auto decoded = decode_cloud(layout, m.data, max_points_, max_bytes_);
    if (!decoded.valid) { reject(decoded.error.c_str()); return; }
    if (!grid_->advance(best->x, best->y, t)) { reject("unsafe grid center or time"); return; }
    grid_->insert(decoded.points, t);
    const auto terrain = grid_->snapshot();
    sensor_msgs::msg::PointCloud2 out;
    out.header = m.header;  // Acquisition time; historical points keep sample_stamp.
    const std::uint16_t endian_marker = 1;
    out.is_bigendian = *reinterpret_cast<const std::uint8_t *>(&endian_marker) == 0;
    sensor_msgs::PointCloud2Modifier modifier(out);
    modifier.setPointCloud2Fields(7, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
      "sample_stamp", 1, sensor_msgs::msg::PointField::FLOAT64,
      "ground_z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "classification", 1, sensor_msgs::msg::PointField::UINT8);
    // Preserve natural alignment for every point, including FLOAT64 timestamps.
    out.point_step = 32;
    modifier.resize(terrain.size());
    out.is_dense = true;
    if (!terrain.empty()) {
      sensor_msgs::PointCloud2Iterator<float> x(out, "x"), y(out, "y"), z(out, "z"),
        intensity(out, "intensity"), ground(out, "ground_z");
      sensor_msgs::PointCloud2Iterator<double> stamp(out, "sample_stamp");
      sensor_msgs::PointCloud2Iterator<std::uint8_t> classification(out, "classification");
      for (const auto & p : terrain) {
        *x = static_cast<float>(p.point.x); *y = static_cast<float>(p.point.y);
        *z = static_cast<float>(p.point.z); *intensity = static_cast<float>(p.clearance);
        *stamp = p.sample_stamp; *ground = static_cast<float>(p.ground_z);
        *classification = static_cast<std::uint8_t>(p.classification);
        if (p.classification == Classification::Unknown) out.is_dense = false;
        ++x; ++y; ++z; ++intensity; ++stamp; ++ground; ++classification;
      }
    }
    publisher_->publish(out);
  }
  std::unique_ptr<RollingGrid> grid_;
  std::string fixed_frame_;
  std::atomic<bool> clock_reset_pending_{false};
  rclcpp::JumpHandler::SharedPtr jump_handler_;
  double last_ros_now_ = -1.0;
  double max_scan_age_{}, max_odom_age_{}, max_skew_{}, future_tolerance_{};
  std::size_t max_points_{}, max_bytes_{};
  std::deque<Odom> odometry_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
};
}  // namespace terrain_analysis

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try { rclcpp::spin(std::make_shared<terrain_analysis::TerrainNode>()); }
  catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("terrain_analysis"), "%s", e.what());
    rclcpp::shutdown(); return 1;
  }
  rclcpp::shutdown(); return 0;
}
