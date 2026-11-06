// SPDX-License-Identifier: BSD-3-Clause
#include "terrain_analysis/rolling_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace terrain_analysis {
namespace {
bool finite_point(const Point & p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
}  // namespace

const char * classification_name(Classification c) {
  switch (c) {
    case Classification::Unknown: return "unknown";
    case Classification::Ground: return "ground";
    case Classification::Obstacle: return "obstacle";
    case Classification::BelowGround: return "below_ground";
  }
  return "unknown";
}

RollingGrid::RollingGrid(GridConfig config) : config_(config) {
  // At most 2M sample slots (64 MB sample payload), even for custom settings.
  // Restrict Z as well on insertion to avoid overflow when subtracting elevations.
  if (!std::isfinite(config.cell_size) || config.cell_size < 0.001 ||
      config.cell_size > 100.0 || config.width < 3 || config.width > 501 ||
      config.width % 2 == 0 || config.max_samples_per_cell == 0 ||
      config.max_samples_per_cell > 4096 || config.min_samples == 0 ||
      config.min_samples > config.max_samples_per_cell ||
      config.width * config.width > 2000000 / config.max_samples_per_cell ||
      !std::isfinite(config.quantile) || config.quantile < 0 || config.quantile > 1 ||
      !std::isfinite(config.decay_seconds) || config.decay_seconds <= 0 ||
      !std::isfinite(config.obstacle_clearance) || config.obstacle_clearance <= 0) {
    throw std::invalid_argument("invalid or excessive rolling grid configuration");
  }
  cells_.resize(config.width * config.width);
}

std::optional<std::int64_t> RollingGrid::coordinate(double v, double resolution) {
  if (!std::isfinite(v)) return std::nullopt;
  const double key = std::floor(v / resolution);
  // Keep ample headroom for window arithmetic and reliable integer conversion.
  if (!std::isfinite(key) || std::abs(key) > 1099511627776.0) return std::nullopt;
  return static_cast<std::int64_t>(key);
}

bool RollingGrid::inside(std::int64_t x, std::int64_t y) const {
  const auto half = static_cast<std::int64_t>(config_.width / 2);
  return stamp_ >= 0 && x >= center_x_ - half && x <= center_x_ + half &&
         y >= center_y_ - half && y <= center_y_ + half;
}

std::size_t RollingGrid::slot(std::int64_t x, std::int64_t y) const {
  const auto w = static_cast<std::int64_t>(config_.width);
  const auto mod = [w](std::int64_t v) { return static_cast<std::size_t>((v % w + w) % w); };
  return mod(y) * config_.width + mod(x);
}

void RollingGrid::estimate(Cell & cell) {
  cell.ground.reset();
  if (cell.samples.size() < config_.min_samples) return;
  std::vector<double> elevations;
  elevations.reserve(cell.samples.size());
  for (const auto & sample : cell.samples) elevations.push_back(sample.point.z);
  // CMU's discrete quantile convention: floor(q*N), capped to N-1.
  const auto rank = std::min(elevations.size() - 1,
    static_cast<std::size_t>(config_.quantile * static_cast<double>(elevations.size())));
  std::nth_element(elevations.begin(), elevations.begin() + static_cast<std::ptrdiff_t>(rank), elevations.end());
  cell.ground = elevations[rank];
}

bool RollingGrid::advance(double x, double y, double stamp) {
  const auto cx = coordinate(x, config_.cell_size);
  const auto cy = coordinate(y, config_.cell_size);
  if (!cx || !cy || !std::isfinite(stamp) || stamp < 0 || stamp < stamp_) return false;
  center_x_ = *cx;
  center_y_ = *cy;
  stamp_ = stamp;
  for (auto & cell : cells_) {
    if (!cell.active) continue;
    if (!inside(cell.x, cell.y)) {
      cell.samples.clear();
      cell.ground.reset();
      cell.active = false;
      cell.next = 0;
      continue;
    }
    const auto old_size = cell.samples.size();
    cell.samples.erase(std::remove_if(cell.samples.begin(), cell.samples.end(),
      [this](const Sample & s) { return stamp_ - s.stamp > config_.decay_seconds; }),
      cell.samples.end());
    if (cell.samples.size() != old_size) {
      // Compact into time order before resetting the full-cell FIFO cursor.
      std::sort(cell.samples.begin(), cell.samples.end(),
        [](const Sample & a, const Sample & b) { return a.stamp < b.stamp; });
      cell.next = 0;
      estimate(cell);
    }
    if (cell.samples.empty()) cell.active = false;
  }
  return true;
}

IngestStats RollingGrid::insert(const std::vector<Point> & points, double stamp) {
  IngestStats stats;
  if (stamp_ < 0 || !std::isfinite(stamp) || stamp != stamp_) return stats;
  stats.accepted_stamp = true;
  std::vector<bool> touched(cells_.size(), false);
  for (const auto & point : points) {
    const auto x = coordinate(point.x, config_.cell_size);
    const auto y = coordinate(point.y, config_.cell_size);
    if (!finite_point(point) || std::abs(point.z) > 1e9 || !x || !y) {
      ++stats.invalid;
      continue;
    }
    if (!inside(*x, *y)) { ++stats.outside; continue; }
    const auto index = slot(*x, *y);
    auto & cell = cells_[index];
    if (!cell.active || cell.x != *x || cell.y != *y) {
      cell.active = true;
      cell.x = *x;
      cell.y = *y;
      cell.samples.clear();
      cell.next = 0;
      cell.ground.reset();
      if (cell.samples.capacity() < config_.max_samples_per_cell)
        cell.samples.reserve(config_.max_samples_per_cell);
    }
    if (cell.samples.size() < config_.max_samples_per_cell) {
      cell.samples.push_back({point, stamp});
    } else {
      cell.samples[cell.next] = {point, stamp};
      cell.next = (cell.next + 1) % config_.max_samples_per_cell;
    }
    touched[index] = true;
    ++stats.accepted;
  }
  for (std::size_t i = 0; i < cells_.size(); ++i) if (touched[i]) estimate(cells_[i]);
  return stats;
}

std::optional<double> RollingGrid::ground_at(double x, double y) const {
  const auto cx = coordinate(x, config_.cell_size);
  const auto cy = coordinate(y, config_.cell_size);
  if (!cx || !cy || !inside(*cx, *cy)) return std::nullopt;
  const auto & cell = cells_[slot(*cx, *cy)];
  if (!cell.active || cell.x != *cx || cell.y != *cy) return std::nullopt;
  return cell.ground;
}

TerrainPoint RollingGrid::classify(const Point & point) const {
  TerrainPoint result;
  result.point = point;
  if (!finite_point(point) || std::abs(point.z) > 1e9) return result;
  const auto ground = ground_at(point.x, point.y);
  if (!ground) return result;
  result.ground_z = *ground;
  result.clearance = point.z - *ground;
  if (result.clearance > config_.obstacle_clearance) result.classification = Classification::Obstacle;
  else if (result.clearance < -config_.obstacle_clearance) result.classification = Classification::BelowGround;
  else result.classification = Classification::Ground;
  return result;
}

std::vector<TerrainPoint> RollingGrid::snapshot() const {
  std::vector<TerrainPoint> result;
  result.reserve(sample_count());
  for (const auto & cell : cells_) {
    if (!cell.active) continue;
    for (const auto & sample : cell.samples) {
      auto point = classify(sample.point);
      point.sample_stamp = sample.stamp;
      result.push_back(point);
    }
  }
  return result;
}

std::size_t RollingGrid::sample_count() const {
  std::size_t result = 0;
  for (const auto & cell : cells_) result += cell.samples.size();
  return result;
}

std::size_t RollingGrid::cell_count() const {
  return static_cast<std::size_t>(std::count_if(cells_.begin(), cells_.end(),
    [](const Cell & c) { return c.active; }));
}

std::size_t RollingGrid::max_sample_count() const {
  return config_.width * config_.width * config_.max_samples_per_cell;
}
}  // namespace terrain_analysis
