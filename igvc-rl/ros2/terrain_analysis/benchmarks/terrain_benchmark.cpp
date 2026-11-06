#include "terrain_analysis/rolling_grid.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
using namespace terrain_analysis;
int main(int argc,char**argv){
 if(argc!=8)return 2;
 int scene=std::stoi(argv[1]),seed=std::stoi(argv[2]),n=std::stoi(argv[3]),load=std::stoi(argv[4]);
 std::string csv=argv[5]; double q=std::stod(argv[6]); int repeats=std::stoi(argv[7]);
 if(scene<0||scene>7||seed<0||n<1||n>1000000||load<0||load>8||!std::isfinite(q)||q<0||q>1||repeats<1||repeats>10000)return 2;
 std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));std::uniform_real_distribution<double>xy(-4,4),u(0,1);std::normal_distribution<double>noise(0,.006);
 std::vector<Point> points;std::vector<int>labels;std::vector<double>truth;
 for(int i=0;i<n;i++){
  double x=xy(rng),y=xy(rng),z=0;int label=0;
  if(scene==1)z=.1*x; if(scene==2||scene==4||scene==7)z=.2*x;
  if(scene==3)z=.35*x;
  if(scene==5)z=.2*std::max(0.,x);
  if(scene==6 && i%30!=0)continue;
  bool obstacle=(scene==4||scene==7)&&std::hypot(x-.5,y)<.6;
  double ground=z;
  if(obstacle && (scene==7||u(rng)<.35)){z+=.4;label=1;}
  points.push_back({x,y,z+noise(rng)});labels.push_back(label);truth.push_back(ground);
 }
 std::atomic<bool>stop=false;std::vector<std::thread>threads;
 for(int i=0;i<load;i++)threads.emplace_back([&]{volatile double a=.1;while(!stop){for(int j=0;j<10000;j++)a=std::sin(a)+.1;}});
 std::vector<double>times;size_t unknown=0,tp=0,fn=0,fp=0,tn=0,below=0,samples=0;double mae=0,grade=0;size_t known=0;
 std::ofstream file;if(csv!="-"){file.open(csv);file<<"x,y,z,truth_ground,truth_obstacle,ground,clearance,prediction\n";}
 for(int repeat=-2;repeat<repeats;repeat++){
  GridConfig cfg;cfg.quantile=q;RollingGrid grid(cfg);
  auto t=std::chrono::steady_clock::now();grid.advance(0,0,1);grid.insert(points,1);auto snapshot=grid.snapshot();
  auto end=std::chrono::steady_clock::now();if(repeat>=0)times.push_back(std::chrono::duration<double,std::milli>(end-t).count());
  if(repeat!=repeats-1)continue;
  samples=snapshot.size();
  double sx=0,sz=0,sxx=0,sxz=0;size_t gnum=0;
  for(size_t i=0;i<points.size();i++){
   auto p=grid.classify(points[i]);int pred=p.classification==Classification::Unknown?-1:(p.classification==Classification::Obstacle?1:0);
   if(pred<0)unknown++;else{known++;mae+=std::abs(p.ground_z-truth[i]);if(labels[i]){if(pred==1)tp++;else fn++;}else{if(pred==1)fp++;else tn++;}}
   if(p.classification==Classification::BelowGround)below++;
   if(pred>=0&&!labels[i]){sx+=points[i].x;sz+=p.ground_z;sxx+=points[i].x*points[i].x;sxz+=points[i].x*p.ground_z;gnum++;}
   if(file && i%std::max(size_t(1),points.size()/4000)==0)file<<points[i].x<<","<<points[i].y<<","<<points[i].z<<","<<truth[i]<<","<<labels[i]<<","<<p.ground_z<<","<<p.clearance<<","<<pred<<"\n";
  }
  double ground_count=static_cast<double>(gnum);
  double denom=ground_count*sxx-sx*sx;if(denom!=0)grade=(ground_count*sxz-sx*sz)/denom;
 }
 stop=true;for(auto&t:threads)t.join();std::sort(times.begin(),times.end());
 auto quant=[&](double p){return times[std::min(times.size()-1,size_t(std::ceil(p*static_cast<double>(times.size()))-1))];};
 std::cout<<"{\"scene\":"<<scene<<",\"seed\":"<<seed<<",\"requested_points\":"<<n<<",\"input_points\":"<<points.size()<<",\"load_threads\":"<<load<<",\"quantile\":"<<q<<",\"repeats\":"<<repeats<<",\"p50_ms\":"<<quant(.5)<<",\"p95_ms\":"<<quant(.95)<<",\"p99_ms\":"<<quant(.99)<<",\"retained_samples\":"<<samples<<",\"unknown\":"<<unknown<<",\"tp\":"<<tp<<",\"fn\":"<<fn<<",\"fp\":"<<fp<<",\"tn\":"<<tn<<",\"below\":"<<below<<",\"ground_mae_m\":"<<(known?mae/static_cast<double>(known):0)<<",\"estimated_grade\":"<<grade<<"}\n";
}
