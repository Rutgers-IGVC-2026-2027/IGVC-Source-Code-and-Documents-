#!/usr/bin/env python3
"""Export one saved Blender scene to a portable, static Gazebo SDF world.
Run with the existing bpy interpreter through scripts/run_blender_cpu.py.
Does not render or modify the source scene.
"""
import argparse,hashlib,json,math,pathlib,re,shutil,xml.etree.ElementTree as ET
import bpy
from mathutils import Matrix
from fix_gazebo_obj_normals import add_normals

def el(parent,tag,text=None,**attrs):
    node=ET.SubElement(parent,tag,attrs)
    if text is not None:node.text=str(text)
    return node

def numbers(values):return ' '.join(f'{float(v):.10g}' for v in values)
def xml(path,root):
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def material_color(mat):
    color=list(mat.diffuse_color) if mat else [.5,.5,.5,1]
    procedural=False
    if mat and mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type=='BSDF_PRINCIPLED':
                color=list(node.inputs['Base Color'].default_value)
                procedural=node.inputs['Base Color'].is_linked
                break
    return color,procedural

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    source=a.scene.resolve();a.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(source));scene=bpy.context.scene
    model_name='igvc_course_6200';modeldir=a.output/'models'/model_name;meshes=modeldir/'meshes';meshes.mkdir(parents=True)
    sdf=ET.Element('sdf',version='1.9');model=el(sdf,'model',name=model_name);el(model,'static','true');link=el(model,'link',name='course')
    depsgraph=bpy.context.evaluated_depsgraph_get();objects=[];triangles=[]
    for idx,obj in enumerate(sorted(scene.objects,key=lambda o:o.name)):
        if obj.type!='MESH':continue
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
        verts=[evaluated.matrix_world@v.co for v in mesh.vertices];groups={}
        for tri in mesh.loop_triangles:
            face=tuple(tri.vertices)
            if evaluated.matrix_world.to_3x3().determinant()<0:face=(face[0],face[2],face[1])
            groups.setdefault(tri.material_index,[]).append(face)
            triangles.append([list(verts[v]) for v in face])
        name=f'{idx:03d}_'+re.sub(r'[^a-zA-Z0-9_]','_',obj.name);class_id=int(obj.get('class_id',0));parts=[]
        for mat_id,faces in sorted(groups.items()):
            part=f'{name}_m{mat_id}';filename=part+'.obj';path=meshes/filename
            with path.open('w') as f:
                f.write('# World-space meters, Z-up; no additional transform\n')
                for v in verts:f.write('v '+numbers(v)+'\n')
                for face in faces:f.write('f '+' '.join(str(v+1) for v in face)+'\n')
            add_normals(path)
            visual=el(link,'visual',name=part+'_visual');geometry=el(visual,'geometry');m=el(geometry,'mesh');el(m,'uri',f'model://{model_name}/meshes/{filename}')
            mat=mesh.materials[mat_id] if mat_id<len(mesh.materials) else None;color,procedural=material_color(mat)
            appearance=el(visual,'material');el(appearance,'ambient',numbers(color));el(appearance,'diffuse',numbers(color));el(appearance,'specular','0.04 0.04 0.04 1')
            # Semantic paint and flat pothole markings stay visual-only.
            collision=class_id not in (2,4)
            if collision:
                col=el(link,'collision',name=part+'_collision');g=el(col,'geometry');m=el(g,'mesh');el(m,'uri',f'model://{model_name}/meshes/{filename}')
            parts.append(dict(file='meshes/'+filename,triangles=len(faces),material=mat.name if mat else None,color=color,procedural_base_color_approximated=procedural,collision=collision))
        objects.append(dict(name=obj.name,export_name=name,class_id=class_id,instance_id=int(obj.pass_index),triangles=sum(len(f) for f in groups.values()),parts=parts))
        evaluated.to_mesh_clear()
    xml(modeldir/'model.sdf',sdf)
    config=ET.Element('model');el(config,'name',model_name);el(config,'version','1.0');el(config,'sdf','model.sdf',version='1.9');el(config,'description','Existing Blender6200 scene exported in meters; static geometry, approximate solid materials.')
    xml(modeldir/'model.config',config)
    world_sdf=ET.Element('sdf',version='1.9');world=el(world_sdf,'world',name='igvc_blender_6200');el(world,'gravity','0 0 -9.81')
    physics=el(world,'physics',name='physics',type='ignored');el(physics,'max_step_size','0.001');el(physics,'real_time_factor','1')
    for file,name in [('gz-sim-physics-system','gz::sim::systems::Physics'),('gz-sim-user-commands-system','gz::sim::systems::UserCommands'),('gz-sim-scene-broadcaster-system','gz::sim::systems::SceneBroadcaster'),('gz-sim-sensors-system','gz::sim::systems::Sensors')]:
        plugin=el(world,'plugin',filename=file,name=name)
        if 'Sensors' in name:el(plugin,'render_engine','ogre2')
    scene_tag=el(world,'scene');el(scene_tag,'ambient','0.45 0.45 0.45 1');el(scene_tag,'background','0.7 0.8 0.95 1');el(scene_tag,'shadows','true')
    sun=el(world,'light',name='sun',type='directional');el(sun,'pose','0 0 10 0 0 0');el(sun,'diffuse','0.8 0.8 0.8 1');el(sun,'specular','0.2 0.2 0.2 1');el(sun,'cast_shadows','true')
    sun_obj=next((o for o in scene.objects if o.type=='LIGHT' and o.data.type=='SUN'),None)
    direction=sun_obj.matrix_world.to_3x3()@__import__('mathutils').Vector((0,0,-1)) if sun_obj else (0,0,-1);el(sun,'direction',numbers(direction))
    include=el(world,'include');el(include,'uri','model://'+model_name)
    camera=scene.camera;transform=camera.matrix_world.copy()
    # Gazebo camera +X forward,+Y left,+Z up; Blender +X right,+Y up,-Z forward.
    conversion=Matrix(((0,-1,0),(0,0,1),(-1,0,0)));rotation=transform.to_3x3()@conversion;euler=rotation.to_euler('XYZ')
    rig=el(world,'model',name='saved_camera');el(rig,'static','true');el(rig,'pose',numbers([*transform.translation,*euler]));sensorlink=el(rig,'link',name='camera_link');sensor=el(sensorlink,'sensor',name='camera',type='camera');el(sensor,'always_on','true');el(sensor,'update_rate','10');el(sensor,'topic','/igvc/camera/image')
    cam=el(sensor,'camera');el(cam,'camera_info_topic','/igvc/camera/camera_info');el(cam,'optical_frame_id','igvc_camera_optical_frame');width=scene.render.resolution_x*scene.render.resolution_percentage//100;height=scene.render.resolution_y*scene.render.resolution_percentage//100
    fx=camera.data.lens/camera.data.sensor_width*width;hfov=2*math.atan(width/(2*fx));el(cam,'horizontal_fov',hfov)
    image=el(cam,'image');el(image,'width',width);el(image,'height',height);el(image,'format','R8G8B8');clip=el(cam,'clip');el(clip,'near',camera.data.clip_start);el(clip,'far',camera.data.clip_end)
    xml(a.output/'igvc_6200.sdf',world_sdf)
    report=dict(source_scene=str(source),source_scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),blender_version=bpy.app.version_string,object_count=len(objects),triangle_count=len(triangles),objects=objects,camera=dict(blender_to_world=list(map(list,transform)),gazebo_to_world_rotation=list(map(list,rotation)),gazebo_pose=[*transform.translation,*euler],image_size=[width,height],fx=fx,horizontal_fov=hfov),limitations=['Procedural shaders, bump maps, Cycles lighting and texture noise are approximated by solid diffuse colors.','Static scene only; no robot or controller. Camera sensor is provided but requires Gazebo rendering support.','Paint and synthetic pothole discs are visual-only; potholes are not physical depressions.','SDF export alone does not establish Gazebo runtime or visual parity.'])
    (a.output/'EXPORT.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'exported-triangles.json').write_text(json.dumps(triangles,separators=(',',':'))+'\n')
    shutil.copyfile(__file__,a.output/'export-source.py')
    launch=a.output/'launch.sh';launch.write_text('#!/bin/sh\nset -eu\nROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexport GZ_SIM_RESOURCE_PATH="$ROOT_DIR/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"\nexec gz sim "$ROOT_DIR/igvc_6200.sdf" "$@"\n');launch.chmod(0o755)
    print(json.dumps({k:report[k] for k in ['object_count','triangle_count','camera','limitations']},indent=2))
if __name__=='__main__':main()
