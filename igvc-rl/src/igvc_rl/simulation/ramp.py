"""Generate corridor or retained-course ramps from a shared visual/contact profile.

Each segment is a tilted box with its top endpoints on the sampled profile.
Native primitives avoid mesh-loader failures and whole-ramp convexification.
"""

import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RampProfile:
    height: float = 0.30
    grade: float = 0.15
    transition: float = 0.80
    deck: float = 1.50
    width: float = 2.0
    start: float = 4.0
    spacing: float = 0.10

    def __post_init__(self):
        values = list(asdict(self).values())
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Ramp dimensions and grade must be finite and positive")
        if self.height / self.grade < self.transition:
            raise ValueError("Height must allow two full slope transitions")
        if self.start < 1.75:
            raise ValueError("Require start >=1.75 m for the y=1 spawn and full robot footprint")
        if self.width <= 1.6 or self.spacing > self.transition / 4:
            raise ValueError("Require width >1.6 m and at least four transition segments")

    @property
    def ascent(self):
        return self.height / self.grade + self.transition

    @property
    def end(self):
        return self.start + 2 * self.ascent + self.deck

    def _rise(self, s):
        t, g, length = self.transition, self.grade, self.ascent
        if s <= 0:
            return 0.0
        if s < t:
            return g * s * s / (2 * t)
        if s <= length - t:
            return g * (s - t / 2)
        if s < length:
            return self.height - g * (length - s) ** 2 / (2 * t)
        return self.height

    def z(self, y):
        s = y - self.start
        if s <= self.ascent:
            return self._rise(s)
        if s <= self.ascent + self.deck:
            return self.height
        return self._rise(self.end - y)

    def samples(self):
        # Include each analytic breakpoint and subdivide; no step at the deck.
        breaks = [0, self.transition, self.ascent - self.transition, self.ascent]
        ys = {self.start + s for s in breaks}
        ys |= {self.end - s for s in breaks}
        ys |= {self.start + self.ascent + self.deck}
        points = []
        ordered = sorted(ys)
        for a, b in zip(ordered, ordered[1:]):
            n = max(1, math.ceil((b - a) / self.spacing))
            points.extend((a + (b - a) * i / n, self.z(a + (b - a) * i / n)) for i in range(n))
        return points + [(self.end, 0.0)]


@dataclass(frozen=True)
class CourseRampPlacement:
    """Seed-6200 tangent frame: world position of the local (0, start) entry."""

    x: float = 0.3234673075
    y: float = 8.7592821575
    yaw: float = 0.053915947

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Placement must be finite")

    def world_xy(self, x, y, start):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * x - s * (y - start), self.y + s * x + c * (y - start)


def verify_course_placement(course_file, profile, placement):
    """Check the unchanged actual follower route, obstacles and lateral footprint.

    A circumscribed robot radius plus 0.10 m tracking allowance is conservative
    for every heading, including wheel footprints straddling entry/exit.
    This geometric check does not establish dynamic tracking or wheel contact.
    """
    import numpy as np
    from shapely.geometry import Polygon

    from igvc_rl.course import CourseSpec
    from igvc_rl.planning.follower import AStarFollower

    course = CourseSpec.load_course_6200(course_file)
    follower = AStarFollower(course)
    route = follower.plan(course.start_pose).points
    c, s = math.cos(placement.yaw), math.sin(placement.yaw)
    local = (route - [placement.x, placement.y]) @ np.array([[c, -s], [s, c]])
    length = profile.end - profile.start
    radius = follower.parameters.circumscribed_radius
    mask = (local[:, 1] >= -radius) & (local[:, 1] <= length + radius)
    crossing = local[mask]
    if len(crossing) < 10 or np.any(np.diff(crossing[:, 1]) <= 0):
        raise ValueError("Ramp must cross a forward contiguous section of the actual route")
    if local[:, 1].min() >= -radius or local[:, 1].max() <= length + radius:
        raise ValueError("Ramp requires a flat route before entry and after exit")
    edge_margin = profile.width / 2 - np.max(abs(crossing[:, 0])) - radius
    footprint = Polygon(
        [
            placement.world_xy(x, y, profile.start)
            for x, y in [
                (-profile.width / 2, profile.start),
                (profile.width / 2, profile.start),
                (profile.width / 2, profile.end),
                (-profile.width / 2, profile.end),
            ]
        ]
    )
    clearance = min(footprint.distance(o.geometry) for o in course.obstacles)
    if edge_margin < 0.10 or clearance < 0.10:
        raise ValueError("Ramp intersects obstacles or lacks planned footprint edge clearance")
    return dict(
        route_points=len(route),
        crossing_points=len(crossing),
        footprint_edge_margin_m=float(edge_margin),
        required_tracking_allowance_m=0.10,
        obstacle_clearance_m=clearance,
        entry_xy=placement.world_xy(0, profile.start, profile.start),
        exit_xy=placement.world_xy(0, profile.end, profile.start),
    )


def _text(parent, name, value):
    ET.SubElement(parent, name).text = str(value)


def _box(world, name, pose, size, color, collision=True):
    model = ET.SubElement(world, "model", name=name)
    _text(model, "static", "true")
    _text(model, "pose", pose)
    link = ET.SubElement(model, "link", name="surface")
    for kind in ["collision", "visual"] if collision else ["visual"]:
        element = ET.SubElement(link, kind, name=name + "_" + kind)
        _text(ET.SubElement(ET.SubElement(element, "geometry"), "box"), "size", size)
        if kind == "visual":
            material = ET.SubElement(element, "material")
            _text(material, "ambient", color)
            _text(material, "diffuse", color)
    return link


def generate_ramp_assets(
    directory,
    baseline_world,
    robot_file,
    profile=None,
    grip=1.0,
    *,
    mode="corridor",
    course_file=None,
    placement=None,
):
    """Write isolated assets; return world, map and 3D robot paths.

    grip is a bounded assumed friction coefficient, not calibrated real traction.
    The existing velocity-actuated drive is retained without torque claims.
    """
    if mode not in ("corridor", "course"):
        raise ValueError("Ramp mode must be corridor or course")
    profile = profile or RampProfile()
    placement = placement or CourseRampPlacement()
    verification = None
    if mode == "course":
        if course_file is None:
            raise ValueError("Course mode requires the retained course metadata")
        verification = verify_course_placement(course_file, profile, placement)
    if not math.isfinite(grip) or not 0 < grip <= 1:
        raise ValueError("Assumed grip must be in (0, 1]")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(baseline_world).getroot()
    world = root.find("world")
    retained = set(world)
    for item in list(world) if mode == "corridor" else []:
        if item.tag in ("include", "model"):
            world.remove(item)
    length = math.ceil(profile.end + 4)
    _box(world, "ramp_ground", f"0 {length / 2} -.1 0 0 0", f"12 {length + 12} .2", ".20 .27 .19 1")
    _box(
        world,
        "approach",
        f"0 {(profile.start - 3) / 2} -.024 0 0 0",
        f"{profile.width} {profile.start + 3} .05",
        ".25 .25 .25 1",
        False,
    )
    _box(
        world,
        "departure",
        f"0 {(profile.end + length + 2) / 2} -.024 0 0 0",
        f"{profile.width} {length + 2 - profile.end} .05",
        ".25 .25 .25 1",
        False,
    )
    model = ET.SubElement(world, "model", name="ramp")
    _text(model, "static", "true")
    link = ET.SubElement(model, "link", name="deck")
    samples = profile.samples()
    for i, ((a, za), (b, zb)) in enumerate(zip(samples, samples[1:])):
        w = profile.width / 2
        angle = math.atan2(zb - za, b - a)
        thickness = 0.10
        # Move the center down along the top normal; exact endpoints remain
        # (a,za) and (b,zb). Neighbors meet flush at their top edges. Lower
        # volumes may overlap, entirely below the drivable surface.
        pose = (
            f"0 {(a + b) / 2 + thickness / 2 * math.sin(angle)} "
            f"{(za + zb) / 2 - thickness / 2 * math.cos(angle)} {angle} 0 0"
        )
        size = f"{profile.width} {math.hypot(b - a, zb - za)} {thickness}"
        for kind in ("collision", "visual"):
            item = ET.SubElement(link, kind, name=f"segment_{i}_{kind}")
            _text(item, "pose", pose)
            _text(ET.SubElement(ET.SubElement(item, "geometry"), "box"), "size", size)
            if kind == "visual":
                material = ET.SubElement(item, "material")
                _text(material, "ambient", ".55 .43 .23 1")
                _text(material, "diffuse", ".55 .43 .23 1")
        # White paint follows the exact same piecewise-linear contact profile.
        for sign in (-1, 1):
            angle = math.atan2(zb - za, b - a)
            _box(
                world,
                f"paint_{i}_{sign}",
                f"{sign * (w - 0.045)} {(a + b) / 2} {(za + zb) / 2 + 0.002} {angle} 0 0",
                f".06 {math.hypot(b - a, zb - za)} .002",
                "1 1 1 1",
                False,
            )
    # Flat approach/departure lane paint.
    for name, a, b in (("entry", -3, profile.start), ("exit", profile.end, length + 2)):
        for sign in (-1, 1):
            _box(
                world,
                f"{name}_line_{sign}",
                f"{sign * (profile.width / 2 - 0.045)} {(a + b) / 2} .002 0 0 0",
                f".06 {b - a} .002",
                "1 1 1 1",
                False,
            )
    side = _box(
        world,
        "side_camera",
        # The retained barrel on +X occludes ascent from that side in course mode.
        f"{-6.5 if mode == 'course' else 6.5} "
        f"{(profile.start + profile.end) / 2} 2.0 0 .17 "
        f"{0 if mode == 'course' else math.pi}",
        ".01 .01 .01",
        "0 0 0 1",
        False,
    )
    sensor = ET.SubElement(side, "sensor", name="side", type="camera")
    _text(sensor, "always_on", "true")
    _text(sensor, "update_rate", "10")
    _text(sensor, "topic", "/ramp/side_image")
    camera = ET.SubElement(sensor, "camera")
    _text(camera, "horizontal_fov", "1.55")
    image = ET.SubElement(camera, "image")
    for key, value in (("width", 960), ("height", 540), ("format", "R8G8B8")):
        _text(image, key, value)
    clip = ET.SubElement(camera, "clip")
    _text(clip, "near", ".05")
    _text(clip, "far", "100")
    if mode == "course":
        # Retain the baseline include and camera exactly; transform only new
        # ramp/paint/side-camera models into the tangent frame. No obstacles move.
        for model in list(world):
            if model in retained:
                continue
            name = model.get("name", "")
            if name in ("ramp_ground", "approach", "departure") or name.startswith(
                ("entry_line", "exit_line")
            ):
                world.remove(model)
                continue
            pose = model.find("pose")
            if pose is None:
                pose = ET.SubElement(model, "pose")
                pose.text = "0 0 0 0 0 0"
            x, y, z, roll, pitch, yaw = map(float, pose.text.split())
            x, y = placement.world_xy(x, y, profile.start)
            pose.text = f"{x} {y} {z} {roll} {pitch} {yaw + placement.yaw}"
    world_file = directory / "ramp.sdf"
    ET.indent(root)
    ET.ElementTree(root).write(world_file, encoding="unicode", xml_declaration=True)
    metadata = {
        "course": {"amplitude_m": 0, "wavelength_m": 28, "phase": 0, "obstacles": []},
        "config": {
            "course": {
                "length_m": length,
                "width_m": profile.width,
                "narrow_width_m": profile.width,
                "line_width_m": 0.09,
            }
        },
    }
    source_course = course_file
    course_file = directory / "course.json"
    if mode == "course":
        shutil.copyfile(source_course, course_file)
    else:
        course_file.write_text(json.dumps(metadata, indent=2) + "\n")
    robot = ET.parse(robot_file)
    robot.find(".//plugin[@name='gz::sim::systems::OdometryPublisher']/dimensions").text = "3"
    for collision in robot.findall(".//link/collision"):
        if any(side in collision.get("name", "") for side in ("left", "right")):
            friction = ET.SubElement(ET.SubElement(collision, "surface"), "friction")
            ode = ET.SubElement(friction, "ode")
            _text(ode, "mu", grip)
            _text(ode, "mu2", grip)
    robot_path = directory / "robot.sdf"
    robot.write(robot_path, encoding="unicode", xml_declaration=True)
    (directory / "profile.json").write_text(
        json.dumps(
            {
                **asdict(profile),
                "mode": mode,
                "placement": asdict(placement) if mode == "course" else None,
                "geometric_verification": verification,
                "dimension_status": "engineering assumptions; not runtime calibrated",
                "end": profile.end,
                "ascent": profile.ascent,
                "grip": grip,
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    )
    return world_file, course_file, robot_path
