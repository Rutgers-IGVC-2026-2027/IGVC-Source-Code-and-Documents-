// SPDX-License-Identifier: BSD-3-Clause
#include "terrain_analysis/rolling_grid.hpp"
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// One bounded frame at a time, no full-sequence retention. Empty frames use an
// odom-only row (three empty point fields). Snapshot contains actual core output.
int main(int argc, char ** argv) {
  if (argc != 3) {
    std::cerr << "Usage: terrain_csv_replay input.csv output.csv\n";
    return 2;
  }
  try {
    if (std::string(argv[1]) == argv[2] ||
        (std::filesystem::exists(argv[2]) && std::filesystem::equivalent(argv[1], argv[2])))
      throw std::runtime_error("input and output must differ");
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("cannot open input");
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot open output");
    std::string line;
    const auto read_line = [&]() {
      line.clear();
      char ch;
      while (input.get(ch)) {
        if (ch == '\n') return true;
        if (line.size() >= 4096) throw std::runtime_error("oversized CSV row");
        line.push_back(ch);
      }
      return !line.empty();
    };
    if (!read_line() || line != "stamp,odom_x,odom_y,x,y,z")
      throw std::runtime_error("expected header stamp,odom_x,odom_y,x,y,z (LF line endings)");
    output << "frame_stamp,sample_stamp,x,y,z,ground_z,clearance,class\n" << std::setprecision(17);
    terrain_analysis::RollingGrid grid;
    std::vector<terrain_analysis::Point> points;
    double stamp = -1, cx = 0, cy = 0;
    const auto flush = [&]() {
      if (stamp < 0) return;
      if (!grid.advance(cx, cy, stamp)) throw std::runtime_error("invalid frame time or odometry");
      const auto stats = grid.insert(points, stamp);
      if (stats.invalid) throw std::runtime_error("invalid or unsafe point coordinate");
      const auto snapshot = grid.snapshot();
      if (snapshot.empty()) output << stamp << ",,,,,,,empty\n";
      for (const auto & p : snapshot) {
        output << stamp << ',' << p.sample_stamp << ',' << p.point.x << ',' << p.point.y << ','
          << p.point.z << ',' << p.ground_z << ',' << p.clearance << ','
          << terrain_analysis::classification_name(p.classification) << '\n';
      }
      if (!output) throw std::runtime_error("output write failed");
      points.clear();
    };
    std::size_t row = 1;
    while (read_line()) {
      ++row;
      if (line.size() > 4096) throw std::runtime_error("oversized CSV row");
      std::vector<std::string> fields;
      std::size_t start = 0;
      for (;;) {
        const auto comma = line.find(',', start);
        fields.push_back(line.substr(start, comma == std::string::npos ? comma : comma - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
      if (fields.size() != 6) throw std::runtime_error("expected six fields at row " + std::to_string(row));
      const auto number = [&fields](std::size_t i) {
        std::size_t used = 0;
        const auto v = std::stod(fields[i], &used);
        if (used != fields[i].size() || !std::isfinite(v)) throw std::runtime_error("invalid CSV number");
        return v;
      };
      const double t = number(0), x = number(1), y = number(2);
      if (t < 0 || t < stamp) throw std::runtime_error("timestamps must be nonnegative and ordered");
      if (t != stamp) { flush(); stamp = t; cx = x; cy = y; }
      else if (x != cx || y != cy) throw std::runtime_error("odometry changed within one frame");
      if (fields[3].empty() && fields[4].empty() && fields[5].empty()) continue;
      if (points.size() >= 200000) throw std::runtime_error("frame exceeds 200000 points");
      points.push_back({number(3), number(4), number(5)});
    }
    if (input.bad()) throw std::runtime_error("input read failed");
    flush();
    output.close();
    if (!output) throw std::runtime_error("output close failed");
  } catch (const std::exception & e) {
    std::cerr << "Replay failed: " << e.what() << '\n'; return 1;
  }
  return 0;
}
