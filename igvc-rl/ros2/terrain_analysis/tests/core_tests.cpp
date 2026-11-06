// SPDX-License-Identifier: BSD-3-Clause
#include "terrain_analysis/cloud_decoder.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace terrain_analysis;
namespace {
void check(bool condition, const char * why) { if (!condition) throw std::runtime_error(why); }
void near(double a, double b, double tolerance = 1e-9) {
  check(std::isfinite(a) && std::abs(a-b) <= tolerance, "value outside expected tolerance");
}
std::vector<Point> repeats(Point p, std::size_t n = 8) { return std::vector<Point>(n, p); }

void ramp() {
  RollingGrid grid;
  check(grid.advance(0, 0, 1), "advance");
  std::vector<Point> points;
  // 5.7 m vertical change across the window. No flattening to odometry Z.
  for (int ix = -48; ix <= 48; ++ix) {
    const double x = static_cast<double>(ix) * 0.2 + 0.05;
    for (int j = 0; j < 8; ++j) {
      const double px = x + static_cast<double>(j) * 0.003;
      points.push_back({px, 0.05, 2.0 + 0.3 * px});
    }
  }
  grid.insert(points, 1);
  const auto result = grid.snapshot();
  check(result.size() == points.size(), "lost ramp points");
  for (const auto & p : result) {
    check(p.classification == Classification::Ground, "ramp classified as obstacle");
    check(std::abs(p.clearance) < 0.01, "ramp clearance too large");
    check(std::abs(p.ground_z - p.point.z) < 0.01, "ramp world elevation lost");
  }
  check(*grid.ground_at(9, .05) - *grid.ground_at(-9, .05) > 5, "ramp flattened");
}
void obstacle() {
  RollingGrid grid;
  grid.advance(0, 0, 1);
  auto points = repeats({.05, .05, 4}, 12);
  const auto objects = repeats({.06, .05, 4.7}, 4);
  points.insert(points.end(), objects.begin(), objects.end());
  grid.insert(points, 1);
  near(*grid.ground_at(.05, .05), 4);
  const auto obstacle = grid.classify({.06, .05, 4.7});
  check(obstacle.classification == Classification::Obstacle, "obstacle missed");
  near(obstacle.clearance, .7);
  check(grid.classify({.05, .05, 3.6}).classification == Classification::BelowGround, "negative clearance lost");
  check(grid.classify({.05, .05, 4.1}).classification == Classification::Ground, "ground threshold");
  GridConfig q; q.min_samples = 4; q.quantile = .25;
  RollingGrid discrete(q); discrete.advance(0, 0, 1);
  discrete.insert({{.01,.01,0},{.01,.01,1},{.01,.01,2},{.01,.01,3}},1);
  near(*discrete.ground_at(.01,.01),1); // floor(.25 * 4)
}
void sparse() {
  RollingGrid grid;
  check(!grid.ground_at(0,0), "uninitialized ground invented");
  grid.advance(0, 0, 1);
  grid.insert(repeats({.05,.05,7},5),1);
  check(!grid.ground_at(.05,.05), "sparse cell declared known");
  for (const auto & p : grid.snapshot()) {
    check(p.classification == Classification::Unknown && std::isnan(p.clearance), "unknown encoding");
  }
  grid.insert({{.05,.05,7}},1);
  near(*grid.ground_at(.05,.05),7);
  check(!grid.ground_at(.25,.05), "neighbor interpolation invented ground");
  check(grid.classify({.25,.05,7}).classification == Classification::Unknown, "unobserved classified free");
}
void rolling() {
  GridConfig c; c.width = 5; c.min_samples = 2;
  RollingGrid grid(c);
  grid.advance(0,0,1);
  grid.insert(repeats({-.35,.05,1}),1);
  grid.insert(repeats({.25,.05,2}),1);
  check(grid.cell_count() == 2, "initial cells");
  grid.advance(.4,0,1.1);
  check(!grid.ground_at(-.35,.05), "evicted cell retained");
  near(*grid.ground_at(.25,.05),2);
  check(!grid.ground_at(.65,.05), "aliased slot inherited ground");
  grid.insert(repeats({.65,.05,3}),1.1);
  near(*grid.ground_at(.65,.05),3);
  grid.advance(-100,-100,1.2);
  check(grid.sample_count() == 0, "teleport retained history");
  grid.advance(0,0,1.3);
  check(!grid.ground_at(.25,.05), "return resurrected history");
  grid.insert(repeats({-.001,-.001,6}),1.3);
  near(*grid.ground_at(-.1,-.1),6);
  check(!grid.ground_at(.001,.001), "negative floor indexing wrong");
}
void expiry() {
  GridConfig c; c.min_samples=4; c.decay_seconds=2;
  RollingGrid grid(c);
  grid.advance(0,0,1); grid.insert(repeats({.05,.05,0},4),1);
  grid.advance(0,0,2); grid.insert(repeats({.05,.05,1},3),2);
  grid.advance(0,0,3); check(grid.ground_at(.05,.05).has_value(), "expiry boundary not inclusive");
  grid.advance(0,0,3.001);
  check(grid.sample_count()==3 && !grid.ground_at(.05,.05), "expired samples supported ground");
  grid.advance(0,0,4.001);
  check(grid.sample_count()==0 && grid.cell_count()==0, "expired history retained");
}
void invalid() {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto inf = std::numeric_limits<double>::infinity();
  RollingGrid grid;
  check(!grid.advance(nan,0,1) && !grid.advance(0,0,nan) && !grid.advance(1e100,0,1), "unsafe advance accepted");
  check(!grid.insert({{0,0,0}},1).accepted_stamp, "insert before center");
  grid.advance(0,0,3); grid.insert(repeats({.05,.05,5}),3);
  const auto before = grid.sample_count();
  check(!grid.advance(100,100,2), "backward clock accepted");
  check(grid.sample_count()==before, "failed advance mutated storage");
  near(*grid.ground_at(.05,.05),5);
  check(!grid.insert({{0,0,0}},4).accepted_stamp, "future sample accepted");
  check(!grid.insert({{0,0,0}},nan).accepted_stamp, "NaN sample time accepted");
  const auto stats = grid.insert({{nan,0,0},{0,inf,0},{0,0,nan},{1e100,0,0},{0,0,1e100},{100,0,0}},3);
  check(stats.invalid==5 && stats.outside==1 && stats.accepted==0, "invalid point accounting");
  check(grid.classify({.05,.05,nan}).classification==Classification::Unknown, "invalid classify");
  for (int i=0;i<8;++i) {
    GridConfig c;
    if (i==0) c.cell_size=nan;
    if (i==1) c.width=104;
    if (i==2) c.max_samples_per_cell=0;
    if (i==3) c.min_samples=65;
    if (i==4) c.quantile=1.01;
    if (i==5) c.decay_seconds=-1;
    if (i==6) { c.width=501; c.max_samples_per_cell=4096; }
    if (i==7) c.obstacle_clearance=inf;
    bool threw=false;
    try { RollingGrid bad(c); } catch (const std::invalid_argument &) { threw=true; }
    check(threw,"invalid config accepted");
  }
}
void decoder() {
  CloudLayout l{1,2,16,20,false,{{"x",0,7,1},{"y",4,7,1},{"z",8,8,1}}};
  std::vector<std::uint8_t> bytes(40,0);
  const auto fill = [&bytes](std::size_t offset, float x, float y, double z) {
    std::memcpy(bytes.data()+offset,&x,4); std::memcpy(bytes.data()+offset+4,&y,4);
    std::memcpy(bytes.data()+offset+8,&z,8);
  };
  const std::uint16_t marker=1;
  l.big_endian=*reinterpret_cast<const std::uint8_t *>(&marker)==0;
  fill(0,1,2,3); fill(20,4,5,6);
  auto result=decode_cloud(l,bytes);
  check(result.valid && result.points.size()==2,"valid organized padded mixed-width cloud rejected");
  near(result.points[1].z,6);
  for (const auto off : {0U,20U}) {
    std::reverse(bytes.begin()+off,bytes.begin()+off+4);
    std::reverse(bytes.begin()+off+4,bytes.begin()+off+8);
    std::reverse(bytes.begin()+off+8,bytes.begin()+off+16);
  }
  l.big_endian=!l.big_endian;
  result=decode_cloud(l,bytes);
  check(result.valid,"opposite endian rejected"); near(result.points[0].x,1); near(result.points[1].z,6);
  for (int i=0;i<10;++i) {
    auto bad=l; auto data=bytes;
    if(i==0) data.pop_back();
    if(i==1) bad.fields.pop_back();
    if(i==2) bad.fields.push_back(bad.fields[0]);
    if(i==3) bad.fields[0].offset=1000;
    if(i==4) bad.fields[0].count=2;
    if(i==5) bad.fields[0].datatype=6;
    if(i==6) bad.row_step=1;
    if(i==7) bad.fields[1].offset=0;
    if(i==8) bad.fields.push_back({"junk",0,99,1});
    if(i==9) bad.width=std::numeric_limits<std::uint32_t>::max();
    check(!decode_cloud(bad,data).valid,"malformed cloud accepted");
  }
  check(!decode_cloud(l,bytes,1).valid,"point limit ignored");
  check(!decode_cloud(l,bytes,10,39).valid,"byte limit ignored");
  l.big_endian=*reinterpret_cast<const std::uint8_t *>(&marker)==0;
  fill(0,std::numeric_limits<float>::quiet_NaN(),2,3); fill(20,4,5,6);
  result=decode_cloud(l,bytes);
  check(result.valid && result.points.size()==1 && result.nonfinite_points==1,"NaN handling failed");
  l.width=0; l.height=1; l.row_step=0;
  check(decode_cloud(l,{}).valid,"well-formed empty cloud rejected");
}
void bounded() {
  GridConfig c; c.width=5; c.max_samples_per_cell=8; c.min_samples=4; c.decay_seconds=100;
  RollingGrid grid(c);
  for(int frame=0;frame<100;++frame) {
    const double t=frame+1;
    grid.advance(frame*.2,0,t);
    std::vector<Point> points;
    for(int x=-2;x<=2;++x) for(int y=-2;y<=2;++y)
      for(int j=0;j<50;++j) points.push_back({frame*.2+x*.2+.01,y*.2+.01,t});
    grid.insert(points,t);
    check(grid.sample_count()<=grid.max_sample_count(),"sample storage unbounded");
    check(grid.cell_count()<=25,"cell storage unbounded");
  }
  // Full-cell FIFO must retain new observations and expire by original stamp.
  RollingGrid fifo(c); fifo.advance(0,0,1); fifo.insert(repeats({.01,.01,0},8),1);
  fifo.advance(0,0,2); fifo.insert(repeats({.01,.01,10},8),2);
  near(*fifo.ground_at(.01,.01),10);
  fifo.advance(0,0,102.001); check(fifo.sample_count()==0,"replacement timestamp wrong");
}
}  // namespace
int main(int argc,char ** argv) {
  try {
    check(argc==2,"supply one test case");
    const std::string name=argv[1];
    if(name=="ramp") ramp(); else if(name=="obstacle") obstacle();
    else if(name=="sparse") sparse(); else if(name=="rolling") rolling();
    else if(name=="expiry") expiry(); else if(name=="invalid") invalid();
    else if(name=="decoder") decoder(); else if(name=="bounded") bounded();
    else throw std::runtime_error("unknown test");
    std::cout << "PASS " << name << '\n';
  } catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
  return 0;
}
