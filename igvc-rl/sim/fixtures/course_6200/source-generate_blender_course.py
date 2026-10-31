"""Render one fixed course and export its evaluated geometry from the same scene.

Use a functioning bpy Python. Outputs are synthetic diagnostics, not real sensor
accuracy. The caller owns authorization and the campaign's cumulative view ledger.
"""
import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def generate(output, config_path, seed=6200, view='front'):
    import bpy
    import numpy as np
    import generate as course
    from lidar_raycast import mesh_from_blender, validate_mesh

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    start = time.monotonic()
    scene, camera, spec = course.build(config, seed)
    course.set_camera(camera, view, spec, config)
    geometry = mesh_from_blender(bpy.context.evaluated_depsgraph_get())
    validate_mesh(geometry)
    # Exported IDs must resolve to the SAME generator instance/class registry.
    for cls, instance in zip(geometry['semantic_ids'], geometry['instance_ids']):
        if int(spec['instances'][str(instance)]['class_id']) != cls:
            raise ValueError('mesh/generator class registry mismatch')
    course.render(scene, camera, spec, config, output, view, config_path)
    (output / 'geometry.json').write_text(json.dumps(geometry, separators=(',', ':'))+'\n')
    for source in (Path(__file__), ROOT/'src/generate.py', ROOT/'src/lidar_raycast.py'):
        shutil.copyfile(source, output / ('source-' + source.name))
    shutil.copyfile(config_path, output/'render-config.json')
    metadata = json.loads((output/'metadata.json').read_text())
    report = {
        'kind': 'actual-blender-course-render-and-evaluated-mesh',
        'blender_version': bpy.app.version_string, 'python': sys.version,
        'machine': platform.machine(), 'seed': seed, 'view': view,
        'resolution': config['resolution'], 'samples': config['samples'],
        'elapsed_seconds': time.monotonic()-start,
        'triangles': len(geometry['triangles_world_m']),
        'instances': len(set(geometry['instance_ids'])),
        'mesh_class_triangle_counts': {str(c):geometry['semantic_ids'].count(c) for c in sorted(set(geometry['semantic_ids']))},
        'world_bounds_m': [np.asarray(geometry['triangles_world_m']).min(axis=(0,1)).tolist(), np.asarray(geometry['triangles_world_m']).max(axis=(0,1)).tolist()],
        'pixel_counts': metadata['pixel_counts'], 'geometry_label_check': metadata['validation'],
        'calibration': metadata['camera'],
        'limitations': ['Synthetic implementation assumptions, no real sensor accuracy.',
                        'Static course only; sequence and offset-sensor alignment are separate gates.'],
        'files': {p.name: {'sha256':course.digest(p), 'bytes':p.stat().st_size}
                  for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'course-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('blender_version','triangles','instances','elapsed_seconds','pixel_counts','geometry_label_check')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=6200)
    parser.add_argument('--view', choices=('front','overview'), default='front')
    args = parser.parse_args()
    generate(args.output, args.config, args.seed, args.view)
