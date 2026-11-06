// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace terrain_analysis {

struct Point { double x{}, y{}, z{}; };

struct GridConfig {
  double cell_size = 0.2;
  std::size_t width = 105;  // Odd, 21 m square at the default resolution.
  std::size_t max_samples_per_cell = 64;
  std::size_t min_samples = 6;
  double quantile = 0.25;
  double decay_seconds = 2.0;
  double obstacle_clearance = 0.15;
};

enum class Classification { Unknown, Ground, Obstacle, BelowGround };
const char * classification_name(Classification value);

struct TerrainPoint {
  Point point;
  double sample_stamp{};
  double ground_z = std::numeric_limits<double>::quiet_NaN();
  double clearance = std::numeric_limits<double>::quiet_NaN();
  Classification classification = Classification::Unknown;
};

struct IngestStats {
  bool accepted_stamp = false;
  std::size_t accepted{}, invalid{}, outside{};
};

// No ROS, PCL, Eigen, global plane fit, or interpolation. Single-threaded API.
class RollingGrid {
public:
  explicit RollingGrid(GridConfig config = {});
  const GridConfig & config() const { return config_; }

  // Invalid/backward timestamps and invalid/unsafe centers fail without mutation.
  // A new time expires old samples; exactly decay_seconds old is still valid.
  bool advance(double center_x, double center_y, double stamp);
  // stamp must exactly match the latest successful advance(). Caller bounds input.
  IngestStats insert(const std::vector<Point> & points, double stamp);
  std::optional<double> ground_at(double x, double y) const;
  TerrainPoint classify(const Point & point) const;
  // Includes observed unknown points with NaN ground/clearance, never empty cells.
  std::vector<TerrainPoint> snapshot() const;
  std::size_t sample_count() const;
  std::size_t cell_count() const;
  std::size_t max_sample_count() const;
  double current_stamp() const { return stamp_; }

private:
  struct Sample { Point point; double stamp; };
  struct Cell {
    bool active = false;
    std::int64_t x{}, y{};
    std::size_t next = 0;
    std::vector<Sample> samples;
    std::optional<double> ground;
  };
  static std::optional<std::int64_t> coordinate(double value, double resolution);
  bool inside(std::int64_t x, std::int64_t y) const;
  std::size_t slot(std::int64_t x, std::int64_t y) const;
  void estimate(Cell & cell);
  GridConfig config_;
  std::vector<Cell> cells_;
  std::int64_t center_x_{}, center_y_{};
  double stamp_ = -1.0;
};
}  // namespace terrain_analysis
