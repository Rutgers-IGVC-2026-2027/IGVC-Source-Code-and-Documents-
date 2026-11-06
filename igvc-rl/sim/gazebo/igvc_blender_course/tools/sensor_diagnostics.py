#!/usr/bin/env python3
"""Record actual acquisition/control messages and bounded smoke commands."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,Image
from nav_msgs.msg import Odometry,OccupancyGrid
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs_py import point_cloud2
from ros_gz_interfaces.msg import Contacts

def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
class Monitor(Node):
 def __init__(self,a):
  super().__init__('sensor_run_diagnostics');self.a=a;self.start=time.monotonic();self.motion_started=None;self.count={};self.last={};self.files=[];self.out=Path(a.output);self.out.mkdir(parents=True,exist_ok=True);self.f=(self.out/'messages.jsonl').open('w');self.files.append(self.f);self.pub=self.create_publisher(Twist,'/cmd_vel',1);self.samples=0;self.last_sample_x=None;self.current_x=0.;self.last_smoke_command=-float("inf")
  self.set_parameters([rclpy.parameter.Parameter('use_sim_time',value=True)])
  for topic in ['/camera/image_raw','/external_camera/image_raw']:self.create_subscription(Image,topic,lambda m,t=topic:self.image(m,t),qos_profile_sensor_data)
  for topic in ['/lidar/points','/registered_scan','/terrain_map']:self.create_subscription(PointCloud2,topic,lambda m,t=topic:self.cloud(m,t),qos_profile_sensor_data)
  self.create_subscription(Odometry,'/ground_truth/odom',self.odom,qos_profile_sensor_data)
  self.create_subscription(OccupancyGrid,'/terrain_occupancy',self.grid,10)
  for wheel in ['front_left_wheel','rear_left_wheel','front_right_wheel','rear_right_wheel']:
   self.create_subscription(Contacts,'/contacts/'+wheel,lambda m,w=wheel:self.log('contact',wheel=w,stamp=stamp(m),contacts=len(m.contacts),upward_normals=[n.z for c in m.contacts for n in c.normals]),qos_profile_sensor_data)
  self.create_subscription(Twist,'/cmd_vel',lambda m:self.log('command',linear=m.linear.x,angular=m.angular.z),10)
  for topic in ['/sensor_controller/status','/sensor_controller/telemetry','/sensor_terrain/telemetry','/terrain/status','/sensor_terrain/diagnostics','/sensor_controller/diagnostics','/dual_lidar/telemetry']:
   self.create_subscription(String,topic,lambda m,t=topic:self.log('text',topic=t,text=m.data),10)
 def log(self,event,**kw):
  self.count[event]=self.count.get(event,0)+1;self.f.write(json.dumps(dict(event=event,receipt_monotonic=time.monotonic(),sim_now=self.get_clock().now().nanoseconds*1e-9,**kw),allow_nan=False)+'\n')
 def image(self,m,t):self.count[t]=self.count.get(t,0)+1;self.log('image',topic=t,stamp=stamp(m),width=m.width,height=m.height,encoding=m.encoding)
 def odom(self,m):
  p=m.pose.pose.position;self.current_x=p.x;q=m.pose.pose.orientation;self.log('odometry',stamp=stamp(m),frame=m.header.frame_id,child_frame=m.child_frame_id,x=p.x,y=p.y,z=p.z,q=[q.x,q.y,q.z,q.w],achieved_linear=m.twist.twist.linear.x,achieved_yaw=m.twist.twist.angular.z)
 def cloud(self,m,t):
  n=m.width*m.height;self.log('cloud',topic=t,stamp=stamp(m),frame=m.header.frame_id,n=n,fields=[x.name for x in m.fields],bytes=len(m.data))
  # Retain original bytes+schema for first few acquisitions, including nonfinite rays.
  if t=='/lidar/points' and self.samples<20 and (self.samples<3 or self.last_sample_x is None or abs(self.current_x-self.last_sample_x)>.75):
   i=self.samples;self.samples+=1;self.last_sample_x=self.current_x;(self.out/f'raw_cloud_{i}.bin').write_bytes(bytes(m.data));(self.out/f'raw_cloud_{i}.json').write_text(json.dumps(dict(stamp=stamp(m),frame=m.header.frame_id,height=m.height,width=m.width,point_step=m.point_step,row_step=m.row_step,is_bigendian=m.is_bigendian,fields=[dict(name=f.name,offset=f.offset,datatype=f.datatype,count=f.count) for f in m.fields]),indent=2))
   arr=point_cloud2.read_points_numpy(m,field_names=('x','y','z'),skip_nans=True).reshape(-1,3)
   arr=arr[np.isfinite(arr).all(axis=1)]
   self.log('cloud_extent',topic=t,min=arr.min(axis=0).tolist() if len(arr) else [],max=arr.max(axis=0).tolist() if len(arr) else [],z_quantiles=np.quantile(arr[:,2],[.05,.5,.95]).tolist() if len(arr) else [])
 def grid(self,m):
  a=np.asarray(m.data);vals,c=np.unique(a,return_counts=True);self.log('map',stamp=stamp(m),frame=m.header.frame_id,resolution=m.info.resolution,width=m.info.width,height=m.info.height,counts={str(int(v)):int(n) for v,n in zip(vals,c)})
 def tick(self):
  if time.monotonic()-self.last_smoke_command < .05:return
  self.last_smoke_command=time.monotonic()
  if self.a.motion_smoke and all(self.count.get(t,0)>=3 for t in ['/camera/image_raw','/external_camera/image_raw']):
   if self.motion_started is None:self.motion_started=time.monotonic();self.log('motion_smoke_start')
   m=Twist()
   if time.monotonic()-self.motion_started<10:m.linear.x=.35;m.angular.z=.12
   self.pub.publish(m)
 def finish(self):
  if self.a.motion_smoke:
   for _ in range(3):self.pub.publish(Twist());time.sleep(.04)
  (self.out/'summary.json').write_text(json.dumps({'counts':self.count,'wall_seconds':time.monotonic()-self.start,'motion_smoke_started':self.motion_started is not None},indent=2))
  for f in self.files:f.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--duration',type=float,default=30);p.add_argument('--motion-smoke',action='store_true');a=p.parse_args();rclpy.init();node=Monitor(a)
 try:
  while rclpy.ok() and time.monotonic()-node.start<a.duration:rclpy.spin_once(node,timeout_sec=.02);node.tick()
 except KeyboardInterrupt:pass
 finally:node.finish();node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
