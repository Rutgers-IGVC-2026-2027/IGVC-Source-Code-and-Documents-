#!/usr/bin/env python3
"""Real ROS pub/sub contract tests; use an isolated domain and existing Jazzy.
Runs a child terrain node with /clock and saves all received output metadata.
No consumer timeout assertion: downstream consumer owns command expiry.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import traceback
import rclpy
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField


def stamp(t):
    sec = int(t)
    return Time(sec=sec, nanosec=round((t-sec)*1e9))


def unpack(m):
    prefix = '>' if m.is_bigendian else '<'
    types = {PointField.FLOAT32:'f', PointField.FLOAT64:'d', PointField.UINT8:'B'}
    return [{f.name: struct.unpack_from(prefix+types[f.datatype], m.data,
             row*m.row_step+col*m.point_step+f.offset)[0] for f in m.fields}
            for row in range(m.height) for col in range(m.width)]


class Harness:
    def __init__(self, binary, root):
        self.root = root
        self.log = (root/'node.log').open('w')
        self.process = subprocess.Popen([binary, '--ros-args', '-p', 'use_sim_time:=true'],
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.n = rclpy.create_node('terrain_contract_test')
        self.messages = []
        self.captured = []
        self.pub = self.n.create_publisher(PointCloud2, '/registered_scan', 10)
        self.odom_pub = self.n.create_publisher(Odometry, '/state_estimation', 10)
        self.clock_pub = self.n.create_publisher(Clock, '/clock', 10)
        self.sub = self.n.create_subscription(PointCloud2, '/terrain_map', self.received, 10)
        try:
            self.spin(1.0)
            deadline=time.monotonic()+6
            while (not self.pub.get_subscription_count() or not self.odom_pub.get_subscription_count()) and time.monotonic()<deadline:
                self.spin(.1)
            assert self.pub.get_subscription_count(), 'terrain node discovery failed'
        except BaseException:
            self.close()
            raise
    def received(self, m):
        self.messages.append(m)
        self.captured.append({'header_stamp':m.header.stamp.sec+m.header.stamp.nanosec/1e9,
            'frame':m.header.frame_id, 'width':m.width, 'height':m.height,
            'point_step':m.point_step, 'is_dense':m.is_dense,
            'fields':[{'name':f.name,'offset':f.offset,'datatype':f.datatype,'count':f.count} for f in m.fields],
            'data_hex':bytes(m.data).hex(),
            'points':[{k:('NaN' if isinstance(v,float) and math.isnan(v) else v) for k,v in p.items()} for p in unpack(m)]})
    def spin(self, duration):
        end=time.monotonic()+duration
        while time.monotonic()<end:
            rclpy.spin_once(self.n, timeout_sec=min(.02,max(0,end-time.monotonic())))
        assert self.process.poll() is None, 'terrain node exited unexpectedly'
    def clock(self,t):
        for _ in range(3):
            self.clock_pub.publish(Clock(clock=stamp(t)))
            self.spin(.035)
    def odom(self,t,frame='map'):
        m=Odometry(); m.header.stamp=stamp(t); m.header.frame_id=frame
        m.pose.pose.orientation.w=1.0
        self.odom_pub.publish(m); self.spin(.08)
    def cloud(self,t,points,frame='map',wide=False,malformed=False):
        m=PointCloud2(); m.header.stamp=stamp(t); m.header.frame_id=frame
        m.height=1; m.width=len(points); m.is_dense=True
        names=['z','x','y'] if wide else ['x','y','z']
        size=8 if wide else 4; m.point_step=size*3
        m.fields=[PointField(name=name,offset=i*size,datatype=8 if wide else 7,count=1) for i,name in enumerate(names)]
        m.is_bigendian=wide
        m.row_step=m.point_step*m.width+(8 if wide and points else 0)
        values=[(p[2],p[0],p[1]) if wide else p for p in points]
        m.data=b''.join(struct.pack('>ddd' if wide else '<fff',*p) for p in values)+bytes(m.row_step-m.point_step*m.width)
        if malformed: m.fields.pop()
        self.pub.publish(m)
    def expect(self,t,points,**kw):
        count=len(self.messages); self.cloud(t,points,**kw)
        deadline=time.monotonic()+1.5
        while len(self.messages)==count and time.monotonic()<deadline: self.spin(.03)
        assert len(self.messages)==count+1, f'expected one output, got {len(self.messages)-count}'
        return self.messages[-1]
    def reject(self,t,points,**kw):
        count=len(self.messages); self.cloud(t,points,**kw); self.spin(.22)
        assert len(self.messages)==count, 'rejected input published/renewed terrain'
    def fresh(self):
        self.clock(1); self.cloud(1,[]); self.spin(.1)
        self.clock(100); self.messages.clear()
    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try: self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGTERM); self.process.wait(timeout=3)
        self.log.close(); self.n.destroy_node()
        (self.root/'messages.json').write_text(json.dumps(self.captured,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--binary',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    rclpy.init(); h=None; results=[]
    try:
        h=Harness(a.binary,root)
        ground=[(.01+i*.01,.01,0) for i in range(6)]
        mixed=ground+[(.08,.01,1)]
        def valid_schema():
            h.odom(100); m=h.expect(100,mixed); rows=unpack(m)
            assert len(rows)==7 and m.header.frame_id=='map' and m.header.stamp==stamp(100)
            assert {f.name:f.datatype for f in m.fields}=={'x':7,'y':7,'z':7,'intensity':7,'sample_stamp':8,'ground_z':7,'classification':2}
            assert m.point_step==32 and m.row_step==32*7 and m.is_dense
            assert sum(p['classification']==2 for p in rows)==1
            assert all(abs(p['intensity']-(p['z']-p['ground_z']))<1e-6 and p['sample_stamp']==100 for p in rows)
        def wide_layout():
            h.odom(100); m=h.expect(100,mixed,wide=True)
            assert len(unpack(m))==7 and sum(p['classification']==2 for p in unpack(m))==1
        def below_ground():
            h.odom(100); m=h.expect(100,ground+[(.08,.01,-1)])
            below=[p for p in unpack(m) if p['classification']==3]
            assert len(below)==1 and below[0]['intensity']==-1 and below[0]['ground_z']==0
        def out_of_order_odom():
            h.odom(100); h.clock(100.2); h.odom(99.9); h.odom(100)
            h.reject(100.2,mixed) # Duplicate/backward odometry cannot become a new match.
            h.odom(100.2); h.expect(100.2,mixed)
        def zero_clock():
            h.clock(0); h.reject(0,mixed); h.odom(0)
            h.clock(10); h.reject(10,mixed)
            h.odom(10); m=h.expect(10,[(1,.01,.5)])
            assert m.width==1 and unpack(m)[0]['sample_stamp']==10
        def wrong_frame():
            h.odom(100); h.reject(100,mixed,frame='sensor')
        def wrong_odom_frame():
            h.odom(100,frame='odom'); h.reject(100,mixed)
        def missing_odom(): h.reject(100,mixed)
        def skew_odom():
            h.odom(99.7); h.reject(100,mixed)
        def stale_odom():
            h.odom(100); h.clock(100.7); h.reject(100.7,mixed)
        def stale_scan():
            h.odom(100); h.clock(100.7); h.odom(100.7); h.reject(100,mixed)
        def future_scan():
            h.odom(100); h.reject(100.2,mixed)
        def duplicate_backward():
            h.odom(100); h.expect(100,mixed); h.clock(100.1)
            h.reject(100,mixed); h.reject(99.9,mixed)
        def malformed():
            h.odom(100); h.reject(100,mixed,malformed=True)
            h.expect(100,mixed) # Rejection must not consume the acquisition stamp.
        def unknown():
            h.odom(100); m=h.expect(100,[(1,.01,.5)]); p=unpack(m)[0]
            assert m.width==1 and not m.is_dense and p['classification']==0
            assert math.isnan(p['intensity']) and math.isnan(p['ground_z']) and p['sample_stamp']==100
        def empty():
            h.odom(100); m=h.expect(100,[]); assert m.width==0 and len(m.data)==0 and len(m.fields)==7
        def historical_expiry():
            h.odom(100); h.expect(100,mixed); h.clock(101); h.odom(101)
            m=h.expect(101,[(1,.01,.5)])
            assert m.header.stamp==stamp(101)
            assert {p['sample_stamp'] for p in unpack(m)}=={100,101}
            h.clock(103.1); h.odom(103.1); m=h.expect(103.1,[])
            assert m.width==0, 'expired points survived'
        def pause_dropout():
            h.odom(100); m=h.expect(100,mixed); count=len(h.messages)
            h.spin(.3); h.clock(101); h.spin(.3)
            assert len(h.messages)==count and m.header.stamp==stamp(100), 'pause renewed old header'
            h.reject(101,mixed) # Clock advanced; previous odometry is stale.
            h.odom(101); m=h.expect(101,mixed)
            assert min(p['sample_stamp'] for p in unpack(m))==100
        def dropout_hidden_clock_reset():
            h.odom(100); h.expect(100,mixed)
            h.clock(150); h.clock(100.5) # Reset ends later than the last input callback.
            h.odom(100.5); m=h.expect(100.5,[(1,.01,.5)])
            assert m.width==1 and unpack(m)[0]['sample_stamp']==100.5, 'dropout hid clock reset; old epoch survived'
        def clock_reset():
            h.odom(100); h.expect(100,mixed)
            h.clock(10); h.reject(10,mixed) # reset clears odometry too
            h.odom(10); m=h.expect(10,[(1,.01,.5)])
            assert m.width==1 and all(p['sample_stamp']==10 for p in unpack(m))
            assert unpack(m)[0]['classification']==0, 'prior epoch ground leaked'
        cases=[valid_schema,wide_layout,below_ground,out_of_order_odom,zero_clock,wrong_frame,wrong_odom_frame,missing_odom,skew_odom,stale_odom,stale_scan,future_scan,duplicate_backward,malformed,unknown,empty,historical_expiry,pause_dropout,dropout_hidden_clock_reset,clock_reset]
        for case in cases:
            start=time.monotonic()
            try:
                h.fresh(); case(); result={'case':case.__name__,'passed':True}
            except Exception:
                result={'case':case.__name__,'passed':False,'traceback':traceback.format_exc()}
            result['wall_seconds']=time.monotonic()-start; results.append(result)
            print(json.dumps(result),flush=True)
        # A large int64 parameter must fail, rather than wrap to valid width 105.
        with (root/'invalid-parameter.log').open('w') as log:
            try:
                check=subprocess.run([a.binary,'--ros-args','-p','grid_width:=4294967401'],
                    stdout=log,stderr=subprocess.STDOUT,timeout=3)
                results.append({'case':'integer_parameter_no_narrowing','passed':check.returncode!=0,'returncode':check.returncode})
            except subprocess.TimeoutExpired:
                results.append({'case':'integer_parameter_no_narrowing','passed':False,'error':'invalid parameter accepted; startup timed out'})
    except Exception:
        results.append({'case':'harness_setup','passed':False,'traceback':traceback.format_exc()})
    finally:
        if h is not None: h.close()
        rclpy.shutdown()
        report={'domain_id':os.environ.get('ROS_DOMAIN_ID'),'tests':results,'passed':all(x['passed'] for x in results) and bool(results),'consumer_timeout':'Not tested here; downstream consumer owns command expiry.'}
        (root/'result.json').write_text(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
