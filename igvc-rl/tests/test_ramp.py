"""Ramp geometry and map contracts independent of a Gazebo installation."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from igvc_rl.course import CourseSpec
from igvc_rl.planning.follower import AStarFollower
from igvc_rl.simulation.ramp import RampProfile, generate_ramp_assets

ROOT = Path(__file__).resolve().parents[1]


def test_profile_contact_grade_and_flush_transitions():
    for grade in (0.10, 0.15):
        p = RampProfile(grade=grade)
        points = np.array(p.samples())
        slopes = np.diff(points[:, 1]) / np.diff(points[:, 0])
        assert np.max(abs(slopes)) <= grade + 1e-9
        assert math.isclose(points[:, 1].max(), 0.30)
        assert points[0, 1] == points[-1, 1] == 0
        assert abs(slopes[0]) < 0.01 and abs(slopes[-1]) < 0.01
        assert p.z(p.end + 1) == p.z(p.start - 1) == 0
        assert p.width > 0.74 + 2 * 0.4
    with pytest.raises(ValueError):
        RampProfile(grade=float("nan"))


def test_shared_primitive_surfaces_and_path_on_deck(tmp_path):
    pkg = ROOT / "sim/gazebo/igvc_blender_course"
    p = RampProfile()
    world, course, robot = generate_ramp_assets(
        tmp_path, pkg / "worlds/igvc_6200.sdf", ROOT / "robot/description/igvc_robot.sdf", p
    )
    link = ET.parse(world).find(".//model[@name='ramp']/link")
    collisions, visuals = link.findall("collision"), link.findall("visual")
    assert len(collisions) == len(visuals) == len(p.samples()) - 1
    for i, (collision, visual) in enumerate(zip(collisions, visuals)):
        assert collision.findtext("geometry/box/size") == visual.findtext("geometry/box/size")
        assert collision.findtext("pose") == visual.findtext("pose")
        _, y, z, roll, _, _ = map(float, collision.findtext("pose").split())
        width, length, thickness = map(float, collision.findtext("geometry/box/size").split())
        assert width == p.width
        for sign, target in zip((-1, 1), p.samples()[i : i + 2]):
            endpoint = [
                y + sign * length / 2 * math.cos(roll) - thickness / 2 * math.sin(roll),
                z + sign * length / 2 * math.sin(roll) + thickness / 2 * math.cos(roll),
            ]
            np.testing.assert_allclose(endpoint, target, atol=1e-10)
    assert ET.parse(robot).findtext(".//dimensions") == "3"
    c = CourseSpec.load_course_6200(course)
    follower = AStarFollower(c)
    trajectory = follower.plan(c.start_pose)
    crossing = trajectory.points[
        (trajectory.points[:, 1] >= p.start) & (trajectory.points[:, 1] <= p.end)
    ]
    assert len(crossing) > 10
    assert max(abs(crossing[:, 0])) + 0.74 / 2 < p.width / 2


@pytest.mark.parametrize(
    "kwargs", [{"width": 1.6}, {"width": 1.59}, {"start": 0.1}, {"start": 1.74}]
)
def test_reject_controller_width_boundary_and_spawn_overlap(kwargs):
    with pytest.raises(ValueError):
        RampProfile(**kwargs)


def test_minimum_approach_and_accepted_width_load_controller(tmp_path):
    pkg = ROOT / "sim/gazebo/igvc_blender_course"
    p = RampProfile(width=1.601, start=1.75)
    world, course, _ = generate_ramp_assets(
        tmp_path, pkg / "worlds/igvc_6200.sdf", ROOT / "robot/description/igvc_robot.sdf", p
    )
    c = CourseSpec.load_course_6200(course)
    assert c.start_pose[1] + 0.9 / 2 < p.start
    assert AStarFollower(c).plan(c.start_pose) is not None
    tree = ET.parse(world)
    for name in ("approach", "departure"):
        model = tree.find(f".//model[@name='{name}']")
        assert not model.findall(".//collision")
        z = float(model.findtext("pose").split()[2])
        thickness = float(model.findtext("link/visual/geometry/box/size").split()[2])
        assert z + thickness / 2 == pytest.approx(0.001)


def test_paint_endpoints_follow_sampled_profile(tmp_path):
    pkg = ROOT / "sim/gazebo/igvc_blender_course"
    p = RampProfile()
    world, _, _ = generate_ramp_assets(
        tmp_path, pkg / "worlds/igvc_6200.sdf", ROOT / "robot/description/igvc_robot.sdf", p
    )
    tree = ET.parse(world)
    for i, endpoints in enumerate(zip(p.samples(), p.samples()[1:])):
        for side in (-1, 1):
            model = tree.find(f".//model[@name='paint_{i}_{side}']")
            assert not model.findall(".//collision")
            x, y, z, roll, _, _ = map(float, model.findtext("pose").split())
            width, length, _ = map(float, model.findtext("link/visual/geometry/box/size").split())
            assert abs(x) + width / 2 < p.width / 2
            for sign, target in zip((-1, 1), endpoints):
                np.testing.assert_allclose(
                    [
                        y + sign * length / 2 * math.cos(roll),
                        z + sign * length / 2 * math.sin(roll),
                    ],
                    [target[0], target[1] + 0.002],
                    atol=1e-10,
                )


def test_course_retains_baseline_and_verifies_actual_path_and_collision_meshes(tmp_path):
    import json

    from shapely.geometry import MultiPoint, Polygon

    from igvc_rl.simulation.ramp import CourseRampPlacement, verify_course_placement

    pkg = ROOT / "sim/gazebo/igvc_blender_course"
    baseline = pkg / "worlds/igvc_6200.sdf"
    metadata = ROOT / "sim/fixtures/course_6200/metadata.json"
    p = RampProfile(**json.loads((pkg / "config/ramp_course_6200.json").read_text()))
    placement = CourseRampPlacement()
    world, course, _ = generate_ramp_assets(
        tmp_path,
        baseline,
        ROOT / "robot/description/igvc_robot.sdf",
        p,
        mode="course",
        course_file=metadata,
    )
    tree = ET.parse(world)
    for index, node in enumerate(ET.parse(baseline).getroot().find("world")):
        retained = tree.getroot().find("world")[index]
        # Includes, baseline camera and world settings survive bytewise XML semantics.
        assert ET.tostring(retained).strip() == ET.tostring(node).strip()
    assert course.read_bytes() == metadata.read_bytes()
    report = verify_course_placement(course, p, placement)
    assert report["footprint_edge_margin_m"] > 0.41
    assert report["obstacle_clearance_m"] > 0.54
    assert report["route_points"] == 657
    footprint = Polygon(
        [
            placement.world_xy(x, y, p.start)
            for x, y in [
                (-p.width / 2, p.start),
                (p.width / 2, p.start),
                (p.width / 2, p.end),
                (-p.width / 2, p.end),
            ]
        ]
    )
    model_dir = pkg / "models/igvc_course_6200"
    for collision in ET.parse(model_dir / "model.sdf").findall(".//collision"):
        uri = collision.findtext("geometry/mesh/uri")
        vertices = np.array(
            [
                list(map(float, row.split()[1:4]))
                for row in (model_dir / "meshes" / Path(uri).name).read_text().splitlines()
                if row.startswith("v ")
            ]
        )
        if vertices[:, 2].max() <= 0.001:  # Retained flat ground is intentionally beneath ramp.
            continue
        assert footprint.distance(MultiPoint(vertices[:, :2]).convex_hull) > 0.53
    assert tree.find(".//model[@name='ramp_ground']") is None
    # The exact shared primitive top surfaces remain correct under the rigid transform.
    ramp_pose = list(map(float, tree.findtext(".//model[@name='ramp']/pose").split()))
    for i, collision in enumerate(tree.findall(".//model[@name='ramp']/link/collision")):
        visual = tree.find(f".//model[@name='ramp']/link/visual[@name='segment_{i}_visual']")
        assert collision.findtext("pose") == visual.findtext("pose")
        assert collision.findtext("geometry/box/size") == visual.findtext("geometry/box/size")
        _, y, z, roll, _, _ = map(float, collision.findtext("pose").split())
        _, length, thickness = map(float, collision.findtext("geometry/box/size").split())
        for sign, target in zip((-1, 1), p.samples()[i : i + 2]):
            local_y = y + sign * length / 2 * math.cos(roll) - thickness / 2 * math.sin(roll)
            actual_xy = (
                ramp_pose[0] - math.sin(ramp_pose[5]) * local_y,
                ramp_pose[1] + math.cos(ramp_pose[5]) * local_y,
            )
            np.testing.assert_allclose(actual_xy, placement.world_xy(0, target[0], p.start))
            assert z + sign * length / 2 * math.sin(roll) + thickness / 2 * math.cos(
                roll
            ) == pytest.approx(target[1])
    # Raised lane paint gets the same tangent transform as the physical ramp.
    for i, endpoints in enumerate(zip(p.samples(), p.samples()[1:])):
        for side in (-1, 1):
            paint = tree.find(f".//model[@name='paint_{i}_{side}']")
            x, y, z, roll, _, yaw = map(float, paint.findtext("pose").split())
            _, length, _ = map(float, paint.findtext("link/visual/geometry/box/size").split())
            for sign, target in zip((-1, 1), endpoints):
                offset = sign * length / 2 * math.cos(roll)
                np.testing.assert_allclose(
                    [x - math.sin(yaw) * offset, y + math.cos(yaw) * offset],
                    placement.world_xy(side * (p.width / 2 - 0.045), target[0], p.start),
                )
                assert z + sign * length / 2 * math.sin(roll) == pytest.approx(target[1] + 0.002)
    camera_xy = list(map(float, tree.findtext(".//model[@name='side_camera']/pose").split()))[:2]
    from shapely.geometry import LineString

    for y in np.linspace(p.start, p.end, 30):
        sightline = LineString([camera_xy, placement.world_xy(0, y, p.start)])
        assert all(
            not sightline.intersects(o.geometry)
            for o in CourseSpec.load_course_6200().physical_obstacles
        )
    with pytest.raises(ValueError):
        verify_course_placement(course, RampProfile(grade=0.1), placement)
    with pytest.raises(ValueError, match="obstacles|clearance"):
        verify_course_placement(course, RampProfile(width=4), placement)
    with pytest.raises(ValueError):
        verify_course_placement(course, p, CourseRampPlacement(x=2.1, y=9))
