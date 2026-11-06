// SPDX-License-Identifier: BSD-3-Clause
#include "terrain_analysis/cloud_decoder.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace terrain_analysis {
namespace {
std::uint32_t field_size(std::uint32_t type) {
  switch (type) {
    case 1: case 2: return 1;
    case 3: case 4: return 2;
    case 5: case 6: case 7: return 4;
    case 8: return 8;
    default: return 0;
  }
}
double read_number(const std::uint8_t * data, const CloudField & f, bool big_endian) {
  const std::uint16_t marker = 1;
  const bool host_big = *reinterpret_cast<const std::uint8_t *>(&marker) == 0;
  std::array<std::uint8_t, 8> bytes{};
  const auto size = field_size(f.datatype);
  std::copy(data + f.offset, data + f.offset + size, bytes.begin());
  if (host_big != big_endian) std::reverse(bytes.begin(), bytes.begin() + size);
  if (f.datatype == 7) { float v; std::memcpy(&v, bytes.data(), 4); return v; }
  double v; std::memcpy(&v, bytes.data(), 8); return v;
}
}  // namespace

DecodeResult decode_cloud(const CloudLayout & l, const std::vector<std::uint8_t> & data,
  std::size_t max_points, std::size_t max_bytes)
{
  DecodeResult result;
  const auto fail = [&result](const char * why) { result.error = why; return result; };
  const auto count = static_cast<std::uint64_t>(l.width) * l.height;
  const auto packed_row = static_cast<std::uint64_t>(l.width) * l.point_step;
  const auto bytes = static_cast<std::uint64_t>(l.height) * l.row_step;
  if (count > max_points || bytes > max_bytes || data.size() > max_bytes)
    return fail("cloud exceeds configured resource limits");
  if (l.height == 0 || l.point_step == 0 || l.row_step < packed_row || bytes != data.size())
    return fail("invalid dimensions, strides, or buffer length");
  std::array<const CloudField *, 3> xyz{{nullptr, nullptr, nullptr}};
  if (l.fields.size() > 128) return fail("too many fields");
  for (const auto & field : l.fields) {
    const auto size = field_size(field.datatype);
    if (size == 0 || field.count == 0 ||
        static_cast<std::uint64_t>(field.offset) +
        static_cast<std::uint64_t>(field.count) * size > l.point_step)
      return fail("invalid field type, count, or offset");
    for (std::size_t i = 0; i < xyz.size(); ++i) {
      if (field.name != std::array<const char *, 3>{{"x", "y", "z"}}[i]) continue;
      if (xyz[i] || field.count != 1 || (field.datatype != 7 && field.datatype != 8))
        return fail("coordinates must be unique scalar FLOAT32 or FLOAT64 fields");
      xyz[i] = &field;
    }
  }
  for (const auto * field : xyz) if (!field) return fail("missing x, y, or z");
  for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = i + 1; j < 3; ++j) {
    const auto a = xyz[i]->offset, b = xyz[j]->offset;
    if (a < b + field_size(xyz[j]->datatype) && b < a + field_size(xyz[i]->datatype))
      return fail("overlapping coordinate fields");
  }
  result.points.reserve(static_cast<std::size_t>(count));
  if (count == 0) { result.valid = true; return result; }
  for (std::uint32_t row = 0; row < l.height; ++row) {
    for (std::uint32_t col = 0; col < l.width; ++col) {
      const auto offset = static_cast<std::size_t>(row) * l.row_step +
        static_cast<std::size_t>(col) * l.point_step;
      const auto * p = data.data() + offset;
      Point point{read_number(p, *xyz[0], l.big_endian),
        read_number(p, *xyz[1], l.big_endian), read_number(p, *xyz[2], l.big_endian)};
      if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
        result.points.push_back(point);
      else ++result.nonfinite_points;
    }
  }
  result.valid = true;
  return result;
}
}  // namespace terrain_analysis
