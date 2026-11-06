"""Join four actually acquired, time-matched simulated 3D scans.

Rigid upper and low-mounted near-field lidars use declared model extrinsics.
This produces a front-frame union of real simulated returns;
it does not create points, surfaces or free cells. Simulation-only geometry.
"""
import json,time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import String

def decode_xyz(m, max_points=100000,max_bytes=8000000):
    n=int(m.width)*int(m.height)
    if n>max_points or len(m.data)>max_bytes or m.point_step<=0 or m.row_step<m.width*m.point_step or len(m.data)<m.row_step*m.height:
        raise ValueError('Invalid/bounded point-cloud layout')
    if n==0:return np.empty((0,3),np.float32)
    names=[];formats=[];offsets=[]
    for name in ('x','y','z'):
        fs=[f for f in m.fields if f.name==name]
        if len(fs)!=1:raise ValueError('Missing/duplicate XYZ field')
        f=fs[0];size={7:4,8:8}.get(f.datatype)
        if size is None or f.count!=1 or f.offset+size>m.point_step:raise ValueError('Invalid XYZ type/offset')
        names.append(name);formats.append(('>' if m.is_bigendian else '<')+('f4' if size==4 else 'f8'));offsets.append(f.offset)
    dtype=np.dtype(dict(names=names,formats=formats,offsets=offsets,itemsize=m.point_step))
    a=np.ndarray((m.height,m.width),dtype=dtype,buffer=m.data,strides=(m.row_step,m.point_step))
    return np.column_stack([a[k].ravel() for k in names]).astype(np.float32)

def union_in_front_frame(front,rear):
    front=np.asarray(front,dtype=np.float32);rear=np.array(rear,dtype=np.float32,copy=True)
    if front.ndim!=2 or rear.ndim!=2 or front.shape[1]!=3 or rear.shape[1]!=3:raise ValueError('Expected XYZ arrays')
    rear[:,0]-=.56
    return np.concatenate([front,rear]),np.r_[np.zeros(len(front),np.uint8),np.ones(len(rear),np.uint8)]

class DualLidar(Node):
    def __init__(self,**kwargs):
        super().__init__('dual_lidar',**kwargs);self.buffers={};self.last_ros=-1;self.last_output=-1;self.counts={};self.pub=self.create_publisher(PointCloud2,'/lidar/points',qos_profile_sensor_data);self.tele=self.create_publisher(String,'~/telemetry',10)
        # Clock notifications catch resets hidden entirely between scan callbacks.
        self.clock_jump_handle = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
            pre_callback=self._invalidate_clock_epoch)
        for which,frame,topic in [('front','lidar_link','/lidar_front/points'),('rear','lidar_rear_link','/lidar_rear/points'),('near_front','near_front_link','/near_front/points'),('near_rear','near_rear_link','/near_rear/points')]:self.create_subscription(PointCloud2,topic,lambda m,w=which,f=frame:self.scan(m,w,f),qos_profile_sensor_data)
    def _invalidate_clock_epoch(self):
        self.buffers.clear()
        self.last_output = -1
        self.last_ros = -1

    def scan(self,m,which,frame):
        start=time.perf_counter();now=self.get_clock().now().nanoseconds
        if self.last_ros>=0 and now<self.last_ros:self.buffers.clear();self.last_output=-1
        self.last_ros=now
        sec,ns=m.header.stamp.sec,m.header.stamp.nanosec;stamp=sec*1000000000+ns
        if sec<0 or ns>=1000000000 or stamp<=0 or m.header.frame_id!=frame or not -50000000<=now-stamp<=500000000:return
        previous=self.buffers.get(which)
        if previous is not None and stamp<=previous[0]:return
        try:xyz=decode_xyz(m)
        except ValueError as e:self.get_logger().error(str(e));return
        self.buffers[which]=(stamp,xyz,m.header)
        if len(self.buffers)<4:return
        sources=[self.buffers[k] for k in ['front','rear','near_front','near_rear']]
        stamps=[x[0] for x in sources];skew=max(stamps)-min(stamps)
        if skew>1000000:return
        oldest=min(stamps)
        if oldest<=self.last_output:return
        offsets=[(0.,0.,0.),(-.56,0.,0.),(.22,0.,-.75),(-.78,0.,-.75)]
        xyz=np.concatenate([x[1]+np.asarray(offset,np.float32) for x,offset in zip(sources,offsets)])
        ids=np.concatenate([np.full(len(x[1]),i,np.uint8) for i,x in enumerate(sources)])
        data=np.zeros(len(xyz),dtype=np.dtype(dict(names=['x','y','z','sensor_id','sample_stamp'],formats=['<f4','<f4','<f4','u1','<f8'],offsets=[0,4,8,12,16],itemsize=24)))
        for j,key in enumerate(['x','y','z']):data[key]=xyz[:,j]
        data['sensor_id']=ids;data['sample_stamp']=np.asarray(stamps,dtype=np.float64)[ids]*1e-9
        out=PointCloud2();out.header=next(x[2] for x in sources if x[0]==oldest);out.header.frame_id='lidar_link';out.height=1;out.width=len(data);out.point_step=24;out.row_step=len(data)*24;out.is_bigendian=False;out.is_dense=bool(np.isfinite(xyz).all());out.fields=[PointField(name=k,offset=j*4,datatype=7,count=1) for j,k in enumerate(['x','y','z'])]+[PointField(name='sensor_id',offset=12,datatype=2,count=1),PointField(name='sample_stamp',offset=16,datatype=8,count=1)];out.data=data.tobytes();self.pub.publish(out);self.last_output=oldest
        self.tele.publish(String(data=json.dumps(dict(acquisition_stamp=oldest*1e-9,match_skew_ms=skew*1e-6,source_points=[len(x[1]) for x in sources],output_points=len(xyz),join_ms=(time.perf_counter()-start)*1000))))

def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);n=DualLidar()
    try:rclpy.spin(n)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:n.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
