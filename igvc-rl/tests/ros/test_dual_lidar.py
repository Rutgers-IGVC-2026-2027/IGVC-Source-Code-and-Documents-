import array
import numpy as np
import pytest
from sensor_msgs.msg import PointCloud2,PointField
from igvc_rl.ros.dual_lidar import decode_xyz,union_in_front_frame

def test_rigid_union_preserves_real_rows_and_does_not_modify_input():
 f=np.array([[1,2,3]],np.float32);r=np.array([[4,5,6]],np.float32);before=r.copy();p,ids=union_in_front_frame(f,r)
 assert len(p)==2 and ids.tolist()==[0,1]
 assert np.array_equal(r,before)
 assert np.allclose(p[1],[3.44,5,6])

def test_decoder_obeys_organized_row_padding_and_endianness():
 m=PointCloud2();m.width=1;m.height=2;m.point_step=12;m.row_step=16;m.is_bigendian=True;m.fields=[PointField(name=k,offset=i*4,datatype=7,count=1) for i,k in enumerate(['x','y','z'])]
 raw=np.zeros(32,np.uint8);raw[:12]=np.frombuffer(np.array([1,2,3],dtype='>f4').tobytes(),np.uint8);raw[16:28]=np.frombuffer(np.array([4,5,6],dtype='>f4').tobytes(),np.uint8);m.data=raw.tobytes()
 assert np.allclose(decode_xyz(m),[[1,2,3],[4,5,6]])
 m.data=m.data[:15]
 with pytest.raises(ValueError):decode_xyz(m)

def test_decoder_rejects_oversized_and_duplicate_fields():
 m=PointCloud2();m.width=100001;m.height=1;m.point_step=12;m.row_step=1200012
 with pytest.raises(ValueError):decode_xyz(m)
 m.width=1;m.row_step=12;m.data=bytes(12);m.fields=[PointField(name='x',offset=0,datatype=7,count=1)]*2
 with pytest.raises(ValueError):decode_xyz(m)
