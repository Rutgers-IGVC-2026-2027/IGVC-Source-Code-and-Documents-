// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "terrain_analysis/rolling_grid.hpp"
#include <string>
#include <vector>

namespace terrain_analysis {
struct CloudField {
  std::string name;
  std::uint32_t offset{}, datatype{}, count{};
};
struct CloudLayout {
  std::uint32_t width{}, height{}, point_step{}, row_step{};
  bool big_endian = false;
  std::vector<CloudField> fields;
};
struct DecodeResult {
  bool valid = false;
  std::string error;
  std::vector<Point> points;
  std::size_t nonfinite_points = 0;
};
// PointCloud2 wire layout without a ROS dependency. No unaligned casts.
DecodeResult decode_cloud(const CloudLayout & layout, const std::vector<std::uint8_t> & data,
  std::size_t max_points = 200000, std::size_t max_bytes = 32000000);
}  // namespace terrain_analysis
