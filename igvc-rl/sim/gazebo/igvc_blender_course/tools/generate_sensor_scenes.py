#!/usr/bin/env python3
"""Generate bounded diagnostic Gazebo scenes; scene truth is never a planner input."""
import argparse, copy, json, math
from pathlib import Path
import xml.etree.ElementTree as E

def box(name, xyz, size, color, pitch=0):
 m=E.Element('model',name=name);E.SubElement(m,'static').text='true';E.SubElement(m,'pose').text=f'{xyz[0]} {xyz[1]} {xyz[2]} 0 {pitch} 0'
 l=E.SubElement(m,'link',name='link')
 for kind in ('collision','visual'):
  k=E.SubElement(l,kind,name=kind);g=E.SubElement(k,'geometry');E.SubElement(E.SubElement(g,'box'),'size').text=' '.join(map(str,size))
  if kind=='visual':
   mat=E.SubElement(k,'material');E.SubElement(mat,'diffuse').text=color;E.SubElement(mat,'ambient').text=color
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--robot',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 robot=E.parse(a.robot).getroot().find('model');base=robot.find("link[@name='base_link']")
 lidar=base.find("sensor[@name='lidar']");lidar.find('pose').text='.28 0 .55 0 0 0';lidar.find('topic').text='/lidar';lidar.find('gz_frame_id').text='lidar_link'
 vertical=lidar.find('lidar/scan/vertical');vertical.find('samples').text='64';vertical.find('min_angle').text='-1.35';vertical.find('max_angle').text='.25'
 lidar.find('lidar/scan/horizontal/samples').text='720';lidar.find('lidar/range/max').text='18'
 rear=copy.deepcopy(lidar);rear.set('name','lidar_rear');rear.find('pose').text='-.28 0 .55 0 0 0';rear.find('topic').text='/lidar_rear';rear.find('gz_frame_id').text='lidar_rear_link';rear.find('update_rate').text='5';lidar.find('update_rate').text='5';base.append(rear)
 for name,x in [('near_front',.50),('near_rear',-.50)]:
  near=copy.deepcopy(lidar);near.set('name',name);near.find('pose').text=f'{x} 0 -.20 0 0 0';near.find('topic').text='/'+name;near.find('gz_frame_id').text=name+'_link';near.find('lidar/scan/horizontal/samples').text='360';near.find('lidar/scan/vertical/samples').text='24';near.find('lidar/scan/vertical/max_angle').text='-.05';base.append(near)
 # Explicit full model pose is required for elevated-terrain registration.
 for plugin in robot.findall('plugin'):
  if 'OdometryPublisher' in plugin.get('name',''):
   dimensions=plugin.find('dimensions')
   if dimensions is None:dimensions=E.SubElement(plugin,'dimensions')
   dimensions.text='3'
 # Prevent simulation command decay from being mistaken for hardware watchdog evidence.
 camera=base.find("sensor[@name='camera']");camera.find('pose').text='.35 0 .55 0 .09 0';camera.find('topic').text='/camera/image_raw'
 # These values are simulation model assumptions, never a physical calibration.
 for dd in robot.findall('plugin'):
  if 'DiffDrive' in dd.get('name',''):
   for key,val in [('max_linear_velocity','4.2'),('min_linear_velocity','-.3'),('max_angular_velocity','1.5'),('min_angular_velocity','-1.5'),('max_linear_acceleration','.8'),('min_linear_acceleration','-1.2')]:
    item=dd.find(key)
    if item is not None:item.text=val
 for link in robot.findall('link'):
  if link.get('name','').endswith('_wheel'):
   collision=link.find('collision').get('name');sensor=E.SubElement(link,'sensor',name='support_contact',type='contact');E.SubElement(sensor,'always_on').text='true';E.SubElement(sensor,'update_rate').text='50';E.SubElement(sensor,'topic').text='/contacts/'+link.get('name');E.SubElement(E.SubElement(sensor,'contact'),'collision').text=collision
 E.indent(robot);E.ElementTree(robot).write(out/'sensor_robot_model.xml',encoding='unicode')
 specs={
 'flat_obstacle':{'goal':[9.,0.],'obstacle':[4.2,0.,.45]},
 'supported_ramp':{'goal':[9.,0.],'ramp':True},
 'ramp_obstacle':{'goal':[9.,0.],'ramp':True,'obstacle':[3.3,.35,.55]},
 'unsupported':{'goal':[9.,0.],'gap':True},
 'dropout':{'goal':[9.,0.]},
 'moved_obstacle':{'goal':[9.,0.],'moving':True},
 'smoke':{'goal':[9.,0.]}}
 for name,spec in specs.items():
  sdf=E.Element('sdf',version='1.9');w=E.SubElement(sdf,'world',name='sensor_diagnostic')
  physics=E.SubElement(w,'physics',name='bounded',type='ignored');E.SubElement(physics,'max_step_size').text='.002';E.SubElement(physics,'real_time_factor').text='1'
  for filename,cls in [('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster'),('sensors','Sensors'),('contact','Contact')]:
   plugin=E.SubElement(w,'plugin',filename=f'gz-sim-{filename}-system',name=f'gz::sim::systems::{cls}')
   if cls=='Sensors':E.SubElement(plugin,'render_engine').text='ogre2'
  light=E.SubElement(w,'light',name='sun',type='directional');E.SubElement(light,'pose').text='0 0 10 0 0 0';E.SubElement(light,'diffuse').text='.9 .9 .9 1';E.SubElement(light,'direction').text='-.5 .3 -1';E.SubElement(light,'cast_shadows').text='true'
  if spec.get('gap'):
   w.append(box('ground_before',(-4,0,-.1),(14,16,.2),'.35 .42 .35 1'));w.append(box('ground_after',(12,0,-.1),(12,16,.2),'.35 .42 .35 1'))
  elif spec.get('ramp'):
   w.append(box('ground_before',(-5,0,-.1),(14,16,.2),'.35 .42 .35 1'))
   # 4m run, 15% grade, followed by observed elevated plateau.
   angle=math.atan(.15);w.append(box('ramp',(4,0,.2),(4/math.cos(angle),8,.2),'.6 .58 .4 1',-angle))
   w.append(box('plateau',(12,0,.5),(12,16,.2),'.35 .42 .35 1'))
  else:w.append(box('ground',(5,0,-.1),(40,30,.2),'.35 .42 .35 1'))
  # Visual ground strips improve motion visibility; no collision/semantic map input.
  for x in range(-1,15):
   m=box(f'ground_marker_{x}',(x,-3,.003),(.08,.6,.005),'.8 .8 .8 1');w.append(m)
  if 'obstacle' in spec:w.append(box('sensed_obstacle',spec['obstacle'],(.8,1.2,.9),'.8 .15 .1 1'))
  if spec.get('moving'):w.append(box('movable_obstacle',(5,4,.45),(.8,1.2,.9),'.8 .15 .1 1'))
  r=copy.deepcopy(robot);E.SubElement(r,'pose').text='0 0 .02 0 0 0';w.append(r)
  m=E.SubElement(w,'model',name='external_camera');E.SubElement(m,'static').text='true';l=E.SubElement(m,'link',name='link');s=E.SubElement(l,'sensor',name='external',type='camera');E.SubElement(s,'pose').text='5 -11 9 0 .68 1.57079632679';E.SubElement(s,'always_on').text='true';E.SubElement(s,'update_rate').text='10';E.SubElement(s,'topic').text='/external_camera/image_raw';E.SubElement(s,'gz_frame_id').text='external_camera';cam=E.SubElement(s,'camera');E.SubElement(cam,'horizontal_fov').text='1.4';im=E.SubElement(cam,'image');E.SubElement(im,'width').text='960';E.SubElement(im,'height').text='540';E.SubElement(im,'format').text='R8G8B8';clip=E.SubElement(cam,'clip');E.SubElement(clip,'near').text='.1';E.SubElement(clip,'far').text='80'
  E.indent(sdf);E.ElementTree(sdf).write(out/f'{name}.sdf',encoding='unicode',xml_declaration=True)
 (out/'scenes.json').write_text(json.dumps({'sensor_extrinsics_base_link':[.28,0,.55,0,0,0],'lidar_vertical_samples':64,'lidar_vertical_fov':[-1.35,.25],'odometry_dimensions':3,'effective_sensor_extrinsics_model_origin':{'front':[.28,0,.85],'rear':[-.28,0,.85],'near_front':[.50,0,.10],'near_rear':[-.50,0,.10]},'simulated_localization':'ground truth assisted full model-origin XYZ/quaternion; child_frame base_link is a label, link offset is .30m','scenarios':specs},indent=2)+'\n')
if __name__=='__main__':main()
