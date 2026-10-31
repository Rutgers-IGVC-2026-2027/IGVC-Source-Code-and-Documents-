#!/usr/bin/env bash
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Source checkout or share install, then ament prefix (lib/<package> install).
if [[ -f "$script_dir/worlds/igvc_6200.sdf" ]]; then
  package_dir="$script_dir"
elif [[ -f "$script_dir/../../share/igvc_blender_course/worlds/igvc_6200.sdf" ]]; then
  package_dir=$(CDPATH= cd -- "$script_dir/../../share/igvc_blender_course" && pwd)
else
  package_dir="$(ros2 pkg prefix --share igvc_blender_course)"
fi
[[ -f "$package_dir/worlds/igvc_6200.sdf" ]] || {
  echo 'Cannot locate igvc_blender_course resources.' >&2; exit 2;
}
if [[ "${ROS_DISTRO:-}" != jazzy ]]; then
  echo 'Source ROS 2 Jazzy before running this check.' >&2
  exit 2
fi
command -v gz >/dev/null
command -v ros2 >/dev/null
export GZ_SIM_RESOURCE_PATH="$package_dir/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export SDF_PATH="$package_dir/models${SDF_PATH:+:$SDF_PATH}"
gz sim --versions
gz sdf -k "$package_dir/models/igvc_course_6200/model.sdf"
gz sdf -k "$package_dir/worlds/igvc_6200.sdf"
ros2 pkg prefix ros_gz_sim
ros2 pkg prefix ros_gz_bridge
echo 'Parser/dependency checks finished. This does not establish camera, contact, or bridge runtime behavior.'
