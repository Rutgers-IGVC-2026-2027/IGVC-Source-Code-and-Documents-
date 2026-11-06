#include "terrain_analysis/rolling_grid.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
using namespace terrain_analysis;
int main(int argc,char**argv){
 if(argc!=5)return 2;int scene=std::stoi(argv[1]),seed=std::stoi(argv[2]);double speed=std::stod(argv[3]);std::ofstream out(argv[4]);out<<"frame,stamp,center_x,input_points,unknown,obstacle_total,tp,fn,unknown_obstacle,fp,ground_mae_m,update_ms,retained_samples\n";
 if(scene<0||scene>2||seed<0||!std::isfinite(speed)||speed<0||speed>4.2||!out)return 2;
 RollingGrid grid;std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));std::uniform_real_distribution<double>u(-4,4),chance(0,1);std::normal_distribution<double>noise(0,.006);
 for(int frame=0;frame<160;frame++){
  double t=1+frame*.05,cx=speed*frame*.05;std::vector<Point>p;std::vector<double>truth;std::vector<bool>obstacle;
  for(int i=0;i<3200;i++){
   double x=cx+u(rng),y=u(rng),g=(scene? .2*x:0);bool obs=scene&&std::hypot(x-5,y)<.6&&(scene==2||chance(rng)<.35);
   p.push_back({x,y,g+(obs?.4:0)+noise(rng)});truth.push_back(g);obstacle.push_back(obs);
  }
  auto start=std::chrono::steady_clock::now();grid.advance(cx,0,t);grid.insert(p,t);auto snapshot=grid.snapshot();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  size_t unknown=0,totalobs=0,tp=0,fn=0,uo=0,fp=0,known=0;double mae=0;
  for(size_t i=0;i<p.size();i++){
   auto pred=grid.classify(p[i]);if(obstacle[i])totalobs++;
   if(pred.classification==Classification::Unknown){unknown++;if(obstacle[i])uo++;continue;}
   known++;mae+=std::abs(pred.ground_z-truth[i]);bool positive=pred.classification==Classification::Obstacle;
   if(obstacle[i]){if(positive)tp++;else fn++;}else if(positive)fp++;
  }
  out<<frame<<","<<t<<","<<cx<<",3200,"<<unknown<<","<<totalobs<<","<<tp<<","<<fn<<","<<uo<<","<<fp<<","<<(known?mae/static_cast<double>(known):0)<<","<<ms<<","<<snapshot.size()<<"\n";
 }
}
