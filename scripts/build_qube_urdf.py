"""Build the URDF of the QUBE-Servo 2 rotary pendulum from its CAD (STEP files) and the Quanser parameters.

    pip install gmsh trimesh numpy                                  # not needed by the training, only by this script
    python scripts/build_qube_urdf.py                               # writes src/Rotary_Pendulum_RL/assets/data/qube_servo2/
    isaaclab -p ../IsaacLab/scripts/tools/convert_urdf.py \
        src/Rotary_Pendulum_RL/assets/data/qube_servo2/qube_servo2.urdf \
        src/Rotary_Pendulum_RL/assets/data/qube_servo2/qube_servo2.usd \
        --fix_base --joint_stiffness 0 --joint_damping 0

The CAD (``step/*.STEP``) and the assembly (``step/Rotary_pendulum.xml``, SolidWorks to Simscape export) are from
https://github.com/vohongquann/rotary-pendulum-matlab, ``hardware/cad_assembly/step``. The assembly was saved with the arm turned
by -177.2 deg and the pendulum shaft by -135.4 deg; this script undoes both, then turns the pendulum about its shaft so that it
hangs down. Zero pose of the URDF: arm angle 0, pendulum angle 0 = hanging down (+-pi = upright).

    base      the QUBE-Servo 2 housing, fixed. Frame: centre of the top face, z up, the motor axis is z.
    Revolute_1  arm joint, axis z, driven by the motor
    arm       hub (``Circle``) and module housing (``Module``)
    Revolute_2  pendulum joint, axis along the arm (pointing away from the motor axis), passive
    pendulum  the aluminium link (``Pendulum``) and the shaft it turns on (``Pivot``)

Masses are the Quanser ones (QUBE-Servo 2 workbook): arm 0.095 kg (hub, module and the metal rod), pendulum link 0.024 kg. Each
part keeps the density of its group, so the mass is spread as the CAD volumes say, and the inertia tensors come from the meshes.
(The 0.0625 / 0.0425 / 0.03 kg of the Simscape data of the MATLAB repo are not used: they give an aluminium pendulum a density of
7080 kg/m^3.)
"""
import re
from pathlib import Path

import gmsh
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "src" / "Rotary_Pendulum_RL" / "assets" / "data" / "qube_servo2"
STEP = OUT / "step"

ARM_MASS_KG = 0.095           # Quanser: rotary arm mass (hub, module and metal rod)
PENDULUM_MASS_KG = 0.024      # Quanser: pendulum link mass
ARM_ANGLE_DEG = -177.19875123974819       # the pose saved in Rotary_pendulum.xml (RevoluteJoint [Base:Circle])
BASE_MASS_KG = 1.217268                   # CAD default (the base is fixed, its mass does not matter)


def rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def transform(rotation, translation):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, translation
    return t


def step_to_mesh(name: str, size_mm: float) -> trimesh.Trimesh:
    """STEP -> triangle mesh in metres (gmsh, OpenCascade kernel)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.merge(str(STEP / f"{name}_Default_sldprt.STEP"))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
    gmsh.model.mesh.generate(2)
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    tags = gmsh.model.mesh.getNodes()[0]
    index = {int(t): i for i, t in enumerate(tags)}
    faces = []
    for dim_tag in gmsh.model.getEntities(2):
        types, _, node_tags = gmsh.model.mesh.getElements(2, dim_tag[1])
        for et, nt in zip(types, node_tags):
            if et == 2:
                faces += [[index[int(n)] for n in tri] for tri in np.array(nt).reshape(-1, 3)]
    gmsh.finalize()
    mesh = trimesh.Trimesh(nodes * 1e-3, np.array(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    return mesh


def assembly_poses() -> dict[str, np.ndarray]:
    """Pose of every part in the frame of the arm at angle 0 (the motor axis is z, the base top face is z = 0), metres."""
    xml = (STEP / "Rotary_pendulum.xml").read_text()
    poses = {}
    undo_arm = rot([0, 0, 1], -np.radians(ARM_ANGLE_DEG))
    for m in re.finditer(r'<Instance name="(\w+)-1".*?<Rotation>([^<]+)</Rotation>\s*<Translation>([^<]+)</Translation>', xml, re.S):
        rotation = np.array(m.group(2).split(), float).reshape(3, 3)
        translation = np.array(m.group(3).split(), float)
        poses[m.group(1)] = transform(undo_arm @ rotation, undo_arm @ translation)
    return poses


def mass_properties(parts: list[tuple[trimesh.Trimesh, np.ndarray, float]]):
    """Mass, centre of mass and inertia about it (link axes) of parts (mesh, pose in the link frame, density)."""
    items = []
    for mesh, pose, density in parts:
        m = mesh.copy()
        m.apply_transform(pose)
        m.density = density
        items.append((m.mass, m.center_mass, m.moment_inertia))
    mass = sum(i[0] for i in items)
    com = sum(i[0] * i[1] for i in items) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in items:
        d = c - com
        inertia += i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def urdf_inertial(mass, com, inertia):
    return (f'    <inertial>\n      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>\n'
            f'      <mass value="{mass:.6f}"/>\n'
            f'      <inertia ixx="{inertia[0, 0]:.4e}" ixy="{inertia[0, 1]:.4e}" ixz="{inertia[0, 2]:.4e}" '
            f'iyy="{inertia[1, 1]:.4e}" iyz="{inertia[1, 2]:.4e}" izz="{inertia[2, 2]:.4e}"/>\n    </inertial>\n')


def rpy(rotation):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rotation).as_euler("xyz")


def urdf_visual(mesh_file, pose, color):
    r, p, y = rpy(pose[:3, :3])
    x = pose[:3, 3]
    return (f'    <visual>\n      <origin xyz="{x[0]:.6f} {x[1]:.6f} {x[2]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}"/>\n'
            f'      <geometry><mesh filename="meshes/{mesh_file}"/></geometry>\n'
            f'      <material name="{color[0]}"><color rgba="{color[1]}"/></material>\n    </visual>\n')


def main():
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    sizes = {"Base": 3.0, "Circle": 1.0, "Module": 1.0, "Pivot": 0.8, "Pendulum": 1.0}
    meshes = {n: step_to_mesh(n, s) for n, s in sizes.items()}
    for n, m in meshes.items():
        m.export(OUT / "meshes" / f"{n.lower()}.stl")
        print(f"{n:9s} watertight {m.is_watertight}  volume {m.volume * 1e6:8.3f} cm^3  faces {len(m.faces)}")

    poses = assembly_poses()
    module, pivot, pendulum, circle = (poses[k] for k in ("Module", "Pivot", "Pendulum", "Circle"))
    assert np.allclose(circle, np.eye(4), atol=1e-6), "the hub must sit on the motor axis"

    # The pendulum shaft: along the module x axis (pointing away from the motor axis), through the pendulum origin.
    axis = module[:3, 0]
    origin = pendulum[:3, 3] * np.array([1.0, 1.0, 1.0])
    origin = np.array([origin[0], 0.0, module[2, 3]])                 # on the shaft line at the pendulum's position
    assert np.allclose(axis, [-1, 0, 0], atol=1e-6)

    # Turn the pivot and the pendulum about the shaft so that the pendulum (its +z, from the shaft to the tip) points down.
    tip = pendulum[:3, :3] @ np.array([0.0, 0.0, 1.0])
    t_perp = tip - np.dot(tip, axis) * axis
    angle = np.arctan2(np.dot(np.cross(t_perp, [0, 0, -1]), axis), np.dot(t_perp, [0, 0, -1]))
    turn = transform(rot(axis, angle), np.zeros(3))
    turn[:3, 3] = origin - turn[:3, :3] @ origin                      # rotation about the line through ``origin``
    pivot_h, pendulum_h = turn @ pivot, turn @ pendulum
    assert np.allclose(pendulum_h[:3, :3] @ [0, 0, 1], [0, 0, -1], atol=1e-6), "the pendulum does not hang down"

    joint = transform(np.eye(3), origin)                              # frame of the pendulum link at joint angle 0
    inv_joint = np.linalg.inv(joint)

    # Densities: the arm group (hub, module, rod) shares one, the pendulum link has its own.
    v_arm = meshes["Circle"].volume + meshes["Module"].volume + meshes["Pivot"].volume
    rho_arm, rho_pendulum = ARM_MASS_KG / v_arm, PENDULUM_MASS_KG / meshes["Pendulum"].volume
    print(f"density of the arm group {rho_arm:.0f} kg/m^3, of the pendulum {rho_pendulum:.0f} kg/m^3 (aluminium: 2700)")

    arm_parts = [(meshes["Circle"], circle, rho_arm), (meshes["Module"], module, rho_arm)]
    pendulum_parts = [(meshes["Pendulum"], inv_joint @ pendulum_h, rho_pendulum), (meshes["Pivot"], inv_joint @ pivot_h, rho_arm)]
    m_arm, c_arm, i_arm = mass_properties(arm_parts)
    m_pen, c_pen, i_pen = mass_properties(pendulum_parts)
    mb, cb, ib = mass_properties([(meshes["Base"], np.eye(4), BASE_MASS_KG / meshes["Base"].volume)])
    print(f"arm link {m_arm:.4f} kg, com {np.round(c_arm * 1000, 2)} mm; pendulum link {m_pen:.4f} kg, com {np.round(c_pen * 1000, 2)} mm")
    print(f"arm (hub + module) inertia about z {i_arm[2, 2]:.3e}; pendulum link inertia about the shaft {i_pen[0, 0]:.3e} kg m^2")

    grey, red = ("grey", "0.65 0.62 0.59 1"), ("red", "0.75 0.08 0.10 1")     # anodised red of the real QUBE-Servo 2 module and pendulum
    urdf = ['<?xml version="1.0"?>\n<!-- Generated by scripts/build_qube_urdf.py from the CAD of github.com/vohongquann/rotary-pendulum-matlab. Do not edit. -->\n',
            '<robot name="qube_servo2">\n',
            '  <link name="base">\n', urdf_inertial(mb, cb, ib), urdf_visual("base.stl", np.eye(4), grey), '  </link>\n',
            '  <joint name="Revolute_1" type="continuous">\n    <parent link="base"/>\n    <child link="arm"/>\n'
            '    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <axis xyz="0 0 1"/>\n    <limit effort="10" velocity="1000"/>\n  </joint>\n',
            '  <link name="arm">\n', urdf_inertial(m_arm, c_arm, i_arm),
            urdf_visual("circle.stl", circle, grey), urdf_visual("module.stl", module, red), '  </link>\n',
            '  <joint name="Revolute_2" type="continuous">\n    <parent link="arm"/>\n    <child link="pendulum"/>\n'
            f'    <origin xyz="{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}" rpy="0 0 0"/>\n'
            f'    <axis xyz="{axis[0]:.0f} {axis[1]:.0f} {axis[2]:.0f}"/>\n    <limit effort="10" velocity="1000"/>\n  </joint>\n',
            '  <link name="pendulum">\n', urdf_inertial(m_pen, c_pen, i_pen),
            urdf_visual("pendulum.stl", inv_joint @ pendulum_h, red), urdf_visual("pivot.stl", inv_joint @ pivot_h, grey),
            '  </link>\n</robot>\n']
    (OUT / "qube_servo2.urdf").write_text("".join(urdf))
    print("wrote", OUT / "qube_servo2.urdf")


if __name__ == "__main__":
    main()
