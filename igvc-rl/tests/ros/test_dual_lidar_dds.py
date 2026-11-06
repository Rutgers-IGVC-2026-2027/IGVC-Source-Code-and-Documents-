"""Real DDS four-stream contracts; simulation clock, isolated domain 86."""
import time
import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField
from igvc_rl.ros.dual_lidar import DualLidar, decode_xyz

FRAMES=['lidar_link','lidar_rear_link','near_front_link','near_rear_link']
TOPICS=['/lidar_front/points','/lidar_rear/points','/near_front/points','/near_rear/points']
class Harness:
 def __init__(self):
  self.ctx=Context();rclpy.init(context=self.ctx)
  self.io=Node('four_stream_test_io',context=self.ctx)
  self.node=DualLidar(context=self.ctx,parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.ex=SingleThreadedExecutor(context=self.ctx);self.ex.add_node(self.io);self.ex.add_node(self.node)
  self.clock=self.io.create_publisher(Clock,'/clock',10)
  self.pubs=[self.io.create_publisher(PointCloud2,t,qos_profile_sensor_data) for t in TOPICS]
  self.out=[];self.io.create_subscription(PointCloud2,'/lidar/points',self.out.append,qos_profile_sensor_data)
  self.spin(.2);assert all(p.get_subscription_count()>0 for p in self.pubs)
  self.set_clock(100.)
 def spin(self,s=.045):
  end=time.monotonic()+s
  while time.monotonic()<end:self.ex.spin_once(timeout_sec=.002)
 def set_clock(self,s):
  c=Clock();c.clock.sec=int(s);c.clock.nanosec=round((s-int(s))*1e9);self.clock.publish(c);self.spin()
 def send(self,i,s=100.,frame=None):
  m=PointCloud2();m.header.frame_id=FRAMES[i] if frame is None else frame;m.header.stamp.sec=int(s);m.header.stamp.nanosec=round((s-int(s))*1e9)
  m.height=1;m.width=2;m.point_step=12;m.row_step=24;m.fields=[PointField(name=k,offset=j*4,datatype=7,count=1) for j,k in enumerate(['x','y','z'])]
  m.data=np.array([[i+1,2,3],[i+2,4,5]],dtype='<f4').tobytes();self.pubs[i].publish(m);self.spin()
 def all(self,s=100.):
  for i in range(4):self.send(i,s)
 def close(self):
  for n in [self.node,self.io]:self.ex.remove_node(n);n.destroy_node()
  self.ex.shutdown();rclpy.shutdown(context=self.ctx)
@pytest.fixture
def h():
 x=Harness()
 try:yield x
 finally:x.close()

def test_four_actual_streams_preserve_all_rows_source_ids_stamps_and_rigid_geometry(h):
 for i,s in enumerate([100.,100.0002,100.0004,100.0006]):h.send(i,s)
 assert len(h.out)==1
 m=h.out[0];assert m.header.frame_id=='lidar_link' and m.header.stamp.sec==100 and m.header.stamp.nanosec==0
 xyz=decode_xyz(m);assert len(xyz)==8
 offsets=np.array([[0,0,0],[-.56,0,0],[.22,0,-.75],[-.78,0,-.75]])
 expected=np.concatenate([np.array([[i+1,2,3],[i+2,4,5]])+offsets[i] for i in range(4)])
 assert np.allclose(xyz,expected)
 raw=np.frombuffer(m.data,dtype=np.dtype(dict(names=['id','stamp'],formats=['u1','<f8'],offsets=[12,16],itemsize=24)))
 assert raw['id'].tolist()==[0,0,1,1,2,2,3,3]
 assert np.allclose(raw['stamp'],np.repeat([100.,100.0002,100.0004,100.0006],2),rtol=0,atol=1e-9)

def test_wrong_frame_does_not_complete_or_refresh_join(h):
 for i in range(3):h.send(i)
 h.send(3,frame='world');assert not h.out
 h.send(3);assert len(h.out)==1

def test_skewed_four_streams_do_not_publish_and_new_matching_set_recovers(h):
 for i in range(3):h.send(i)
 h.send(3,100.002);assert not h.out
 h.set_clock(100.1);h.all(100.1);assert len(h.out)==1

def test_duplicate_and_backwards_acquisitions_do_not_republish(h):
 h.all();assert len(h.out)==1
 h.all();h.all(99.9);assert len(h.out)==1

def test_dropout_has_no_republication_and_requires_all_fresh_streams(h):
 h.all();assert len(h.out)==1
 h.set_clock(100.7)
 for i in range(3):h.send(i,100.7)
 assert len(h.out)==1
 h.send(3,100.);assert len(h.out)==1
 h.send(3,100.7);assert len(h.out)==2

def test_clock_reset_discards_pre_reset_partials_and_accepts_new_epoch(h):
 for i in range(3):h.send(i)
 h.set_clock(10.);h.send(3,10.);assert not h.out
 for i in range(3):h.send(i,10.)
 assert len(h.out)==1 and h.out[0].header.stamp.sec==10

def test_stale_and_future_sources_do_not_complete_join(h):
 for i in range(3):h.send(i)
 h.send(3,99.4);h.send(3,100.1);assert not h.out
 h.send(3);assert len(h.out)==1


def test_hidden_backward_clock_jump_invalidates_partial_join_and_recovers(h):
 for i in range(3):h.send(i,100.)
 h.set_clock(150.)
 h.set_clock(100.0005)
 h.send(3,100.0005)
 assert not h.out, 'Old epoch partials must not complete a new epoch join'
 for i in range(3):h.send(i,100.0005)
 assert len(h.out)==1
 assert h.out[0].header.stamp.sec==100
 assert h.out[0].header.stamp.nanosec==500000
