#!/usr/bin/env python3
"""Finite, Gazebo-only camera cart motion. No velocity, ROS publisher or hardware."""
import argparse
import math
import os
import re
import subprocess
import sys
import time

WORLD = 'igvc_vehicle_pov_6200'
MODEL = 'igvc_pov_vehicle'
X, Y, YAW = -0.226459071, 0.0, 1.44331944


def request(distance):
    x, y = X + distance * math.cos(YAW), Y + distance * math.sin(YAW)
    return (f'name: "{MODEL}", position: {{x: {x:.9f}, y: {y:.9f}, z: 0}}, '
            f'orientation: {{x: 0, y: 0, z: {math.sin(YAW/2):.12f}, w: {math.cos(YAW/2):.12f}}}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reset', action='store_true', help='one pose request back to start')
    parser.add_argument('--dry-run', action='store_true', help='print endpoints; no Gazebo calls')
    args = parser.parse_args()
    if args.dry_run:
        print(request(0)); print(request(2.0))
        return 0
    if not os.environ.get('GZ_PARTITION', '').startswith('igvc_pov_'):
        parser.error('Set the same isolated igvc_pov_ GZ_PARTITION as the POV launch')
    if os.environ.get('GZ_IP') != '127.0.0.1':
        parser.error('POV requires GZ_IP=127.0.0.1')
    # Fixed absolute path: repeating this program never accumulates displacement.
    # 0.1 m per acknowledged request; >=0.15 wall seconds between requests.
    # Slow service calls slow motion; never catch up or send a backlog.
    distances = [0.0] if args.reset else [i * 0.1 for i in range(21)]
    deadline = time.monotonic() + 30.0
    try:
        for distance in distances:
            if time.monotonic() >= deadline:
                raise RuntimeError('30 second wall deadline exceeded')
            result = subprocess.run(
                ['gz', 'service', '-s', f'/world/{WORLD}/set_pose',
                 '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '500', '--req', request(distance)],
                capture_output=True, text=True, timeout=1.0, check=True)
            if not re.search(r'\bdata:\s*true\b', result.stdout):
                raise RuntimeError('Gazebo did not acknowledge pose: ' + result.stdout.strip())
            time.sleep(0.15)
    except KeyboardInterrupt:
        print('Interrupted; no continuing velocity exists. A pending pose may still apply.')
        return 130
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f'POV stopped: {exc}. No further requests will be sent.', file=sys.stderr)
        return 1
    print('POV complete; cart remains stationary at its last pose.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
