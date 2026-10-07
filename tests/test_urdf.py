"""The URDF of the QUBE-Servo 2 against the Quanser workbook (no simulator)."""
import xml.etree.ElementTree as ET

import numpy as np

from Rotary_Pendulum_RL.rotary import rotary_cfg as C

URDF = C._URDF_PATH


def _robot():
    root = ET.parse(URDF).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    return links, joints


def _inertia(link):
    inertial = link.find("inertial")
    a = inertial.find("inertia")
    tensor = np.array([[a.get(k) for k in ("ixx", "ixy", "ixz")], [a.get("ixy"), a.get("iyy"), a.get("iyz")],
                       [a.get("ixz"), a.get("iyz"), a.get("izz")]], float)
    return float(inertial.find("mass").get("value")), tensor


def test_joints_and_links_are_the_ones_of_the_config():
    links, joints = _robot()
    assert set(links) == {"base", "arm", "pendulum"}
    assert C.ARM_JOINT in joints and C.PENDULUM_JOINT in joints
    assert joints[C.ARM_JOINT].find("axis").get("xyz") == "0 0 1"
    assert joints[C.PENDULUM_JOINT].find("axis").get("xyz") == "-1 0 0"      # along the arm, away from the motor axis


def test_masses_add_up_to_the_workbook():
    links, _ = _robot()
    assert abs(_inertia(links["arm"])[0] + _inertia(links["pendulum"])[0] - (0.095 + 0.024)) < 1e-5


def test_geometry_is_the_one_of_the_workbook():
    """Arm axis to pendulum shaft 85 mm and pendulum 129 mm (QUBE-Servo 2 workbook), within the 1 % of a CAD that is not the
    manufacturer's."""
    _, joints = _robot()
    x, y, z = (float(v) for v in joints[C.PENDULUM_JOINT].find("origin").get("xyz").split())
    assert abs(np.hypot(x, y) - 0.085) < 0.001 and abs(z - 0.0165) < 1e-6
    pendulum_stl = URDF.parent / "meshes" / "pendulum.stl"
    assert pendulum_stl.exists()


def test_inertias_are_physical():
    links, _ = _robot()
    for name in ("base", "arm", "pendulum"):
        mass, tensor = _inertia(links[name])
        assert mass > 0 and np.all(np.linalg.eigvalsh(tensor) > 0), name
        eigen = np.linalg.eigvalsh(tensor)
        assert eigen[-1] <= eigen[0] + eigen[1] + 1e-12, name                # triangle inequality of a rigid body
