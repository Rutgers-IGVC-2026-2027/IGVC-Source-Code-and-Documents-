#!/usr/bin/env python3
"""Add flat face normals to exported triangle OBJs without changing geometry."""
import argparse,hashlib,json,math
from pathlib import Path

def add_normals(path):
    path=Path(path);old=path.read_bytes();lines=old.decode().splitlines()
    vertices=[tuple(map(float,l.split()[1:4])) for l in lines if l.startswith('v ')]
    faces=[tuple(int(i.split('/')[0]) for i in l.split()[1:]) for l in lines if l.startswith('f ')]
    normals=[]
    for face in faces:
        if len(face)!=3 or any(i<1 or i>len(vertices) for i in face):raise ValueError(f'Invalid triangle: {path}')
        a,b,c=(vertices[i-1] for i in face);u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
        n=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);length=math.sqrt(sum(x*x for x in n))
        if not math.isfinite(length) or length<1e-15:raise ValueError(f'Degenerate triangle: {path}')
        normals.append(tuple(x/length for x in n))
    output=[l for l in lines if not l.startswith(('vn ','f '))]
    output.extend('vn '+' '.join(f'{v:.10g}' for v in n) for n in normals)
    output.extend('f '+' '.join(f'{v}//{i}' for v in f) for i,f in enumerate(faces,1))
    new=('\n'.join(output)+'\n').encode();path.write_bytes(new)
    return {'file':path.name,'triangles':len(faces),'normals':len(normals),'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest(),'vertices_and_faces_unchanged':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    results=[add_normals(f) for f in sorted(a.directory.glob('*.obj'))]
    a.report.write_text(json.dumps({'reason':'Gazebo DART rejects collision submeshes without vertex normals; add face normals with explicit OBJ indices.','files':results},indent=2)+'\n')
    print(json.dumps({'files':len(results),'triangles':sum(r['triangles'] for r in results)}))
