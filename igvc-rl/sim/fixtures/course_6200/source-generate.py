"""Deterministic Blender IGVC scene segments and visibility-aware semantic masks.
Run with Python + bpy, or blender -b --python src/generate.py -- [arguments].
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Matrix
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: 'background', 1: 'asphalt', 2: 'boundary', 3: 'barrel', 4: 'pothole', 5: 'environment'}
PALETTE = np.array([[22,32,49],[82,93,105],[255,204,64],[255,103,79],[72,206,235],[92,156,99]], dtype=np.uint8)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def material(name, color, roughness=.65, noise=False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bs = nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, 1)
    bs.inputs['Roughness'].default_value = roughness
    if noise:
        geo = nodes.new('ShaderNodeTexCoord')
        tex = nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 220 if name == 'Asphalt' else 35
        tex.inputs['Detail'].default_value = 3
        links.new(geo.outputs['Object'], tex.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .18
        ramp.color_ramp.elements[0].color = (*[v*(.45 if name in ['Asphalt','Grass'] else .88) for v in color],1)
        ramp.color_ramp.elements[1].position = .82
        ramp.color_ramp.elements[1].color = (*[min(v*(1.7 if name in ['Asphalt','Grass'] else 1.12),1) for v in color],1)
        links.new(tex.outputs['Fac'], ramp.inputs[0])
        links.new(ramp.outputs[0], bs.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .52 if name == 'Asphalt' else .35
        bump.inputs['Distance'].default_value = .004 if name == 'Asphalt' else .003
        links.new(tex.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs[0], bs.inputs['Normal'])
    return mat

def label(obj, cls, mat=None):
    obj.pass_index = cls
    obj['class_id'] = cls
    if mat: obj.data.materials.append(mat)
    return obj

def mesh(name, verts, faces, cls, mat):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return label(obj, cls, mat)

def plane(name, size, loc, mat, cls):
    x,y = size
    return mesh(name, [(loc[0]+a*x/2,loc[1]+b*y/2,loc[2]) for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]], [(0,1,2,3)],cls,mat)

def cylinder(name, loc, radius, depth, mat, cls, vertices=64):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    return label(obj,cls,mat)

def barrel(name, center, radius, height, body, white, dark):
    # Lathed non-lipped drum with modest moulded ribs; no oversized rubber foot.
    levels = [(0,.98),(.035,1.03),(.08,.99),(.22,.95),(.24,.98),(.26,.98),(.28,.94),(.51,.91),(.53,.94),(.55,.94),(.57,.9),(.78,.87),(.80,.9),(.82,.9),(.84,.86),(.95,.82),(1,.80)]
    n=64
    verts=[(center[0]+radius*r*math.cos(2*math.pi*j/n),center[1]+radius*r*math.sin(2*math.pi*j/n),height*z+.004) for z,r in levels for j in range(n)]
    faces=[]
    bands=[]
    for k in range(len(levels)-1):
        z=(levels[k][0]+levels[k+1][0])/2
        for j in range(n):
            faces.append((k*n+j,k*n+(j+1)%n,(k+1)*n+(j+1)%n,(k+1)*n+j))
            bands.append(2 if z>.95 else 1 if .28<z<.51 or .57<z<.78 else 0)
    faces.extend([tuple(reversed(range(n))),tuple((len(levels)-1)*n+j for j in range(n))])
    bands.extend([2,2])
    obj=mesh(name,verts,faces,3,body)
    obj.data.materials.append(white);obj.data.materials.append(dark)
    for poly,band in zip(obj.data.polygons,bands):
        poly.material_index=band
        poly.use_smooth=len(poly.vertices)==4
    return obj

def path_frame(y, spec):
    a,w,p=spec['amplitude_m'],spec['wavelength_m'],spec['phase']
    k=2*math.pi/w
    x=a*math.sin(k*y+p)
    dx=a*k*math.cos(k*y+p)
    norm=math.sqrt(1+dx*dx)
    return x, (1/norm,-dx/norm)

def width_at(y,c):
    # Smooth constriction centered well beyond the first obstacle.
    return c['width_m']-(c['width_m']-c['narrow_width_m'])*math.exp(-((y-18)/3.7)**2)

def add_asphalt_cracks(config,rng,spec):
    """Add a few narrow, low-contrast road cracks; they retain the asphalt label."""
    bounds=config.get('road_cracks_count_range',[0,0])
    count=rng.randint(int(bounds[0]),int(bounds[1]))
    if not count:return
    crack_mat=material('Asphalt crack',(.014,.015,.016),.98)
    c=config['course']
    for i in range(count):
        y=rng.uniform(.8,c['length_m']-1.2)
        center_x,normal=path_frame(y,spec)
        tangent=(-normal[1],normal[0])
        lateral=rng.uniform(-(width_at(y,c)/2-.25),width_at(y,c)/2-.25)
        center=(center_x+normal[0]*lateral,y+normal[1]*lateral)
        length=rng.uniform(.28,.95);half_width=rng.uniform(.003,.007)
        segments=rng.randint(4,7);verts=[];faces=[]
        for j in range(segments+1):
            along=length*(j/segments-.5)
            kink=rng.uniform(-.025,.025)
            cx=center[0]+tangent[0]*along+normal[0]*kink
            cy=center[1]+tangent[1]*along+normal[1]*kink
            width=half_width*rng.uniform(.55,1.35)
            verts.extend([(cx+normal[0]*width,cy+normal[1]*width,.0035),
                          (cx-normal[0]*width,cy-normal[1]*width,.0035)])
            if j:faces.append((2*j-2,2*j,2*j+1,2*j-1))
        mesh(f'Asphalt crack {i:02d}',verts,faces,1,crack_mat)

def add_old_lot_markings(config,rng,spec):
    """Place faded parking-lot paint outside the course, labeled as asphalt."""
    bounds=config.get('lot_markings_count_range',[0,0])
    count=rng.randint(int(bounds[0]),int(bounds[1]))
    if not count:return
    mats=[material('Faded yellow lot paint',(.30,.25,.10),.94,True),
          material('Faded white lot paint',(.34,.34,.30),.94,True)]
    c=config['course']
    for i in range(count):
        y=rng.uniform(3,c['length_m']-2)
        _,normal=path_frame(y,spec);tangent=(-normal[1],normal[0])
        side=-1 if rng.random()<.5 else 1
        start_offset=width_at(y,c)/2+.3
        length=rng.uniform(1.8,3.2)
        # Slight wear breaks a parking stripe into a few irregular sections.
        verts=[];faces=[];position=0.0
        half_width=rng.uniform(.025,.045)
        while position<length:
            section=rng.uniform(.32,.9);gap=rng.uniform(.01,.08)
            lo=position;hi=min(position+section,length)
            for along in (lo,hi):
                lateral=side*(start_offset+along)
                x=path_frame(y,spec)[0]+normal[0]*lateral
                yy=y+normal[1]*lateral
                verts.extend([(x+tangent[0]*half_width,yy+tangent[1]*half_width,.003),
                              (x-tangent[0]*half_width,yy-tangent[1]*half_width,.003)])
            base=len(verts)-4
            face=(base,base+1,base+3,base+2)
            faces.append(face if side>0 else tuple(reversed(face)))
            position=hi+gap
        mesh(f'Worn lot marking {i:02d}',verts,faces,1,rng.choice(mats))

def assign_instance_indices():
    """Give each labeled object a unique visible-instance ID for the IndexOB pass."""
    objects = sorted((obj for obj in bpy.data.objects if 'class_id' in obj), key=lambda obj: obj.name)
    if len(objects) > 255:
        raise ValueError(f'{len(objects)} labeled objects exceed the 8-bit instance-mask limit')
    instance_map = {}
    for instance_id, obj in enumerate(objects, 1):
        cls = int(obj['class_id'])
        obj.pass_index = instance_id
        instance_map[str(instance_id)] = {'name': obj.name, 'class_id': cls}
    return instance_map

def sample_camera_value(rng, camera, name, fallback):
    bounds = camera.get(name + '_range')
    if bounds is not None:
        return rng.uniform(float(bounds[0]), float(bounds[1]))
    return float(camera.get(name, fallback))

def build(config,seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng=random.Random(seed)
    c=config['course']
    spec={'seed':seed,'amplitude_m':rng.uniform(*c['amplitude_m']), 'wavelength_m':rng.uniform(*c['wavelength_m']), 'phase':rng.uniform(-.25,.25)}
    camera=config['camera']
    spec['camera']={
        'height_m':sample_camera_value(rng,camera,'height_m',1.1),
        'focal_length_mm':sample_camera_value(rng,camera,'focal_length_mm',22.0),
        'front_y_m':sample_camera_value(rng,camera,'front_y_m',0.0),
        'lateral_offset_m':sample_camera_value(rng,camera,'lateral_offset_m',0.0),
        'look_distance_m':sample_camera_value(rng,camera,'look_distance_m',9.0),
    }
    if 'pitch_down_deg_range' in camera:
        spec['camera']['pitch_down_deg']=sample_camera_value(rng,camera,'pitch_down_deg',20.0)
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC'
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=config['samples']
    scene.cycles.seed=seed
    scene.cycles.use_denoising=True
    scene.render.resolution_x,scene.render.resolution_y=config['resolution']
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False
    scene.render.use_file_extension=True
    scene.view_settings.view_transform='AgX'
    scene.view_layers[0].use_pass_object_index=True
    asphalt=material('Asphalt',(.20,.17,.135),.91,True)
    grass=material('Grass',(.13,.20,.065),.94,True)
    white=material('White tape',(.82,.83,.78),.5,True)
    reflective=material('Reflective bands',(.88,.90,.85),.32)
    dark=material('Dark plastic',(.025,.029,.032),.64)
    colors=[(.92,.14,.012),(.68,.022,.017),(.02,.18,.58),(.02,.38,.12),(.92,.62,.02)]
    rng.shuffle(colors)
    bodies=[material('Drum '+str(i),col,.4,True) for i,col in enumerate(colors)]
    plane('Surroundings',(220,220),(0,20,-.045),grass,5)
    plane('Asphalt pad',(26,52),(0,16,0),asphalt,1)
    # Add roughness-scale asphalt patches with a second low-frequency texture.
    nodes=asphalt.node_tree.nodes;links=asphalt.node_tree.links
    noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=.42
    coords=nodes.get('Texture Coordinate');links.new(coords.outputs['Object'],noise.inputs['Vector'])
    ramp=nodes.new('ShaderNodeMapRange');ramp.inputs['From Min'].default_value=0;ramp.inputs['From Max'].default_value=1
    ramp.inputs['To Min'].default_value=.65;ramp.inputs['To Max'].default_value=.98
    links.new(noise.outputs['Fac'],ramp.inputs['Value']);links.new(ramp.outputs[0],nodes.get('Principled BSDF').inputs['Roughness'])
    # Large-scale weathering multiplies the fine aggregate color.
    fine=nodes.get('Color Ramp')
    weather=nodes.new('ShaderNodeValToRGB')
    weather.color_ramp.elements[0].color=(.42,.45,.48,1)
    weather.color_ramp.elements[1].color=(1,1,1,1)
    links.new(noise.outputs['Fac'],weather.inputs[0])
    mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.7
    links.new(fine.outputs['Color'],mix.inputs[1]);links.new(weather.outputs['Color'],mix.inputs[2])
    links.new(mix.outputs[0],nodes.get('Principled BSDF').inputs['Base Color'])
    # Seeded shallow asphalt repairs add visible surface variation without changing labels.
    repair_count=int(config.get('road_repairs_count',0))
    repair_mat=material('Asphalt repair',(.16,.145,.12),.94,True) if repair_count else None
    for i in range(repair_count):
        # Coplanar overlapping repairs give Cycles and ray_cast ambiguous instance IDs.
        # Separate them within a 0.5 mm band, still below the 2 mm boundary strips.
        repair_z = .001 + .0005 * i / max(1, repair_count - 1)
        y=rng.uniform(1,c['length_m']-1)
        center_x,normal=path_frame(y,spec)
        lateral=rng.uniform(-(width_at(y,c)/2-.35),width_at(y,c)/2-.35)
        center=(center_x+normal[0]*lateral,y+normal[1]*lateral)
        tangent=(-normal[1],normal[0])
        half_length=rng.uniform(.25,.9)
        half_width=rng.uniform(.10,.30)
        verts=[]
        for j in range(8):
            angle=2*math.pi*j/8
            radial=rng.uniform(.72,1.16)
            along=half_length*math.cos(angle)*radial
            across=half_width*math.sin(angle)*radial
            verts.append((center[0]+tangent[0]*along+normal[0]*across,
                          center[1]+tangent[1]*along+normal[1]*across,repair_z))
        mesh(f'Asphalt repair {i:02d}',verts,[tuple(range(8))],1,repair_mat)
    add_asphalt_cracks(config,rng,spec)
    add_old_lot_markings(config,rng,spec)
    # Flat strips follow the local normal; width is measured perpendicular to centerline.
    for side in [-1,1]:
        verts=[];faces=[]
        ys=np.linspace(-5,c['length_m'],500)
        for y in ys:
            x,n=path_frame(float(y),spec)
            for edge in [-1,1]:
                offset=side*width_at(float(y),c)/2+edge*c['line_width_m']/2
                verts.append((x+n[0]*offset,float(y)+n[1]*offset,.002))
        for j in range(len(ys)-1): faces.append((2*j,2*j+1,2*j+3,2*j+2))
        mesh('Boundary '+str(side),verts,faces,2,white)
    obstacles=[]
    for i,y0 in enumerate(c['barrel_y_m']):
        y=y0+rng.uniform(-.55,.55)
        x,n=path_frame(y,spec)
        width=width_at(y,c)
        side=(-1 if i%2==0 else 1)
        offset=side*(width/2-c['barrel_radius_m']-.28)
        xy=(x+n[0]*offset,y+n[1]*offset)
        barrel('Barrel_%02d'%i,xy,c['barrel_radius_m'],c['barrel_height_m'],bodies[i%len(bodies)],reflective,dark)
        free_width=width/2+abs(offset)-c['barrel_radius_m']-c['line_width_m']/2
        assert free_width>=c['minimum_clearance_m']
        obstacles.append({'kind':'barrel','center_m':[*xy,0],'radius_m':c['barrel_radius_m'],'height_m':c['barrel_height_m'],'opposite_side_clearance_m':free_width})
    for i,y in enumerate(c['pothole_y_m']):
        x,n=path_frame(y,spec); offset=(-.62 if i%2 else .62)
        xy=(x+n[0]*offset,y+n[1]*offset)
        cylinder('Pothole_%02d'%i,(*xy,.0016),c['pothole_diameter_m']/2,.002,white,4,96)
        obstacles.append({'kind':'pothole','center_m':[*xy,0],'diameter_m':c['pothole_diameter_m']})
    # Distant context provides horizon and cast shadows without external assets.
    bark=material('Bark',(.12,.075,.032),.9,True)
    leaves=material('Leaves',(.07,.16,.033),.9,True)
    for i in range(24):
        side=-1 if i%2 else 1
        x=side*rng.uniform(15,23);y=rng.uniform(-7,60);h=rng.uniform(3.8,6.5)
        cylinder('Tree trunk', (x,y,h/3), .13,h*2/3,bark,5,12)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,radius=1,location=(x,y,h*.8))
        tree=bpy.context.object;tree.scale=(1.8,1.6,h*.35);label(tree,5,leaves)
        for poly in tree.data.polygons: poly.use_smooth=True
    # Nishita sky + direct sun, with seed-controlled elevation and orientation.
    scene.world=bpy.data.worlds.new('Sky');scene.world.use_nodes=True
    wn=scene.world.node_tree.nodes;wl=scene.world.node_tree.links
    sky=wn.new('ShaderNodeTexSky');sky.sky_type='NISHITA';sky.sun_disc=False
    lighting=config['lighting']
    elevation=rng.uniform(*lighting['sun_elevation_rad']);rotation=rng.uniform(*lighting['sun_rotation_rad'])
    sky.sun_elevation=elevation;sky.sun_rotation=rotation
    wn.get('Background').inputs['Strength'].default_value=lighting['sky_strength']
    wl.new(sky.outputs['Color'],wn.get('Background').inputs['Color'])
    bpy.ops.object.light_add(type='SUN',location=(0,0,10))
    sun=bpy.context.object;sun.name='Sun';sun.rotation_euler=(math.pi/2-elevation,0,rotation)
    sun.data.energy=rng.uniform(*lighting['sun_energy'])
    angle_bounds=lighting.get('sun_angle_rad_range',[.07,.07])
    sun.data.angle=rng.uniform(*angle_bounds)
    spec.update({'sun_elevation_rad':elevation,'sun_rotation_rad':rotation,'sun_energy':sun.data.energy,'sun_angle_rad':sun.data.angle,'obstacles':obstacles})
    spec['instances']=assign_instance_indices()
    bpy.ops.object.camera_add()
    cam=bpy.context.object;cam.name='Camera';scene.camera=cam
    cam.data.lens=spec['camera']['focal_length_mm'];cam.data.sensor_width=config['camera']['sensor_width_mm'];cam.data.sensor_fit='HORIZONTAL'
    cam.data.clip_start=.05;cam.data.clip_end=300
    return scene,cam,spec

def set_camera(cam, view, spec, config):
    if view=='front':
        pose=spec['camera'];y=pose['front_y_m']
        x,normal=path_frame(y,spec);look_y=y+pose['look_distance_m'];tx,_=path_frame(look_y,spec)
        cam.location=(x+normal[0]*pose['lateral_offset_m'],y+normal[1]*pose['lateral_offset_m'],pose['height_m'])
        target_z=.25
        if 'pitch_down_deg' in pose:
            horizontal_distance=math.hypot(tx-cam.location.x,look_y-cam.location.y)
            target_z=cam.location.z-math.tan(math.radians(pose['pitch_down_deg']))*horizontal_distance
        target=(tx,look_y,target_z)
    elif view=='overview':
        cam.location=(11,-7,14);target=(0,14,0)
    else: raise ValueError('Unknown camera view: '+view)
    cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()
    bpy.context.view_layer.update()

def calibration(cam,config):
    w,h=config['resolution'];f=w*cam.data.lens/cam.data.sensor_width
    # Continuous image coordinates: upper-left image corner (0,0); pixel centers at +0.5.
    world_to_cv=Matrix.Diagonal((1,-1,-1,1))@cam.matrix_world.inverted()
    return {'K':[[f,0,w/2],[0,f,h/2],[0,0,1]],'world_to_camera_cv':[list(r) for r in world_to_cv], 'camera_to_world_blender':[list(r) for r in cam.matrix_world], 'convention':'CV x right, y down, z forward; world meters, Z up; image origin upper-left corner, pixel centers at (u+0.5,v+0.5)', 'distortion':[0,0,0,0,0]}

def render(scene,cam,spec,config,folder,view,config_path):
    folder.mkdir(parents=True,exist_ok=False)
    set_camera(cam,view,spec,config)
    scene.use_nodes=True
    nodes=scene.node_tree.nodes;nodes.clear()
    rl=nodes.new('CompositorNodeRLayers')
    comp=nodes.new('CompositorNodeComposite');scene.node_tree.links.new(rl.outputs['Image'],comp.inputs[0])
    out=nodes.new('CompositorNodeOutputFile');out.base_path='//'
    out.format.file_format='OPEN_EXR';out.format.color_depth='32';out.format.color_mode='RGB'
    out.file_slots[0].path='index_'
    scene.node_tree.links.new(rl.outputs['IndexOB'],out.inputs[0])
    scene.render.filepath='//rgb.png'
    scene.render.use_stamp_filename=False
    # Keep the saved UI's file browser independent of the creator's home directory.
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type=='FILE_BROWSER' and space.params:
                    space.params.directory=b'//'
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'scene.blend'))
    bpy.ops.render.render(write_still=True)
    raw=next(folder.glob('index_*.exr'))
    im=bpy.data.images.load(str(raw),check_existing=False)
    w,h=im.size
    pix=np.asarray(im.pixels[:],dtype=np.float32).reshape(h,w,im.channels)
    ids=np.flipud(pix[:,:,0]).copy()
    if not np.allclose(ids,np.rint(ids),atol=1e-5): raise ValueError('Noninteger object-index pass')
    instance_ids=np.rint(ids).astype(np.uint8)
    if not set(np.unique(instance_ids)).issubset({0}|{int(k) for k in spec['instances']}): raise ValueError('Unknown instance ID')
    class_lookup=np.zeros(256,dtype=np.uint8)
    for instance_id,entry in spec['instances'].items(): class_lookup[int(instance_id)]=int(entry['class_id'])
    mask=class_lookup[instance_ids]
    Image.fromarray(instance_ids).save(folder/'instance.png')
    Image.fromarray(mask).save(folder/'semantic.png')
    (folder/'instances.json').write_text(json.dumps(spec['instances'],indent=2)+'\n')
    Image.fromarray(PALETTE[mask]).save(folder/'semantic_preview.png')
    rgb=np.array(Image.open(folder/'rgb.png').convert('RGB'))
    overlay=np.rint(rgb*.55+PALETTE[mask]*.45).astype(np.uint8)
    Image.fromarray(overlay).save(folder/'overlay.png')
    # Sparse independent center-ray check proves orientation and occlusion semantics.
    deps=bpy.context.evaluated_depsgraph_get();K=calibration(cam,config)['K'];f=K[0][0]
    checked=class_mismatches=instance_mismatches=0
    for v in range(8,h-8,17):
        for u in range(8,w-8,17):
            patch=mask[v-2:v+3,u-2:u+3]
            instance_patch=instance_ids[v-2:v+3,u-2:u+3]
            if np.any(patch!=mask[v,u]) or np.any(instance_patch!=instance_ids[v,u]):
                continue # center-ray checks use class- and instance-interior pixels
            direction=cam.matrix_world.to_3x3()@Vector(((u+.5-w/2)/f,-(v+.5-h/2)/f,-1))
            hit,loc,normal,face,obj,matrix=scene.ray_cast(deps,cam.location,direction.normalized(),distance=cam.data.clip_end)
            expected_class=int(obj.get('class_id',0)) if hit else 0
            expected_instance=obj.pass_index if hit else 0
            checked+=1
            class_mismatches+=int(expected_class!=int(mask[v,u]))
            instance_mismatches+=int(expected_instance!=int(instance_ids[v,u]))
    if class_mismatches or instance_mismatches:
        raise ValueError(f'Ray/mask alignment failed: class={class_mismatches}/{checked}, instance={instance_mismatches}/{checked}')
    counts={CLASSES[k]:int(np.count_nonzero(mask==k)) for k in CLASSES}
    instance_counts={str(i):int(np.count_nonzero(instance_ids==int(i))) for i in spec['instances']}
    meta={'schema_version':2,'seed':spec['seed'],'view':view,'resolution':[w,h],'blender':bpy.app.version_string,'renderer':'Cycles CPU','samples':scene.cycles.samples,'course':spec,'camera':calibration(cam,config),'class_map':CLASSES,'instance_map':spec['instances'],'pixel_counts':counts,'instance_pixel_counts':instance_counts,'validation':{'interior_center_rays_checked':checked,'mismatches':class_mismatches,'instance_mismatches':instance_mismatches},'config':config,'source_sha256':digest(__file__),'config_sha256':digest(config_path),'files':{p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    bpy.data.images.remove(im)
    print('FRAME_COMPLETE',folder.name,json.dumps(counts),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/milestone.json')
    parser.add_argument('--output',type=Path,default=ROOT/'samples')
    parser.add_argument('--seeds',type=int,nargs='+',help='Optional seed override')
    parser.add_argument('--views',nargs='+',choices=['front','overview'])
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    config=json.loads(args.config.read_text());args.output=args.output.resolve()
    seeds=args.seeds if args.seeds is not None else config['seeds'];views=args.views or config['views']
    if len(seeds)!=len(set(seeds)) or len(views)!=len(set(views)):raise ValueError('Duplicate seeds/views')
    for seed in seeds:
        for view in views:
            if (args.output/f'{seed}_{view}').exists():raise FileExistsError('Refusing to overwrite frame directory')
    for seed in seeds:
        scene,cam,spec=build(config,seed)
        for view in views:render(scene,cam,spec,config,args.output/f'{seed}_{view}',view,args.config)
    (args.output/'classes.json').write_text(json.dumps({'classes':CLASSES,'palette':PALETTE.tolist(),'mask_format':'8-bit grayscale PNG, pixel value is class ID; 0 is background, not ignore','instance_mask':'8-bit grayscale PNG; 0 is background and positive IDs map through each frame metadata.instance_map'},indent=2)+'\n')

if __name__=='__main__': main()
