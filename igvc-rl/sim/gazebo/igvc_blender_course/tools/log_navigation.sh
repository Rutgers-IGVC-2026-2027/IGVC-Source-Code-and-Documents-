#!/usr/bin/env bash
# Record only the small topics needed to reconstruct a navigation run.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_BAG_DIRECTORY" >&2
  exit 2
fi

share="$(ros2 pkg prefix --share igvc_blender_course)"
exec ros2 bag record --output "$1" --max-cache-size 1048576 \
  --qos-profile-overrides-path "$share/config/navigation_log_qos.yaml" \
  /clock /cmd_vel /ground_truth/odom /astar_controller/status
