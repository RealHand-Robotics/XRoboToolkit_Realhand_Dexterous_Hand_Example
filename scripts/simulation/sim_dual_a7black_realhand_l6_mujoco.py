"""MuJoCo simulation for dual A7black arms with dual RealHand L6 hands.

The scene is generated at runtime from:
  - assets/real_hand/A7black/dual_A7black.urdf
  - assets/real_hand/l6/left/realhand_l6_left.urdf
  - assets/real_hand/l6/right/realhand_l6_right.urdf

The A7black URDF provides the arm kinematics. This script adds MuJoCo position
actuators for the 14 arm joints, attaches one L6 hand to each A7black tool
link, and adds position actuators for the six independent L6 control joints on
each hand.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import mujoco
from mujoco import viewer as mj_viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"

A7BLACK_DUAL_URDF = ASSETS / "real_hand" / "A7black" / "dual_A7black.urdf"
HAND_URDFS = {
    "left": ASSETS / "real_hand" / "l6" / "left" / "realhand_l6_left.urdf",
    "right": ASSETS / "real_hand" / "l6" / "right" / "realhand_l6_right.urdf",
}

A7_ARM_JOINTS = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
A7_JOINT4_FORWARD_RAD = 0.3490658503988659
A7_READY_JOINTS_RAD = {
    "left_joint2": 0.4363323129985824,
    "left_joint4": A7_JOINT4_FORWARD_RAD,
    "right_joint2": -0.4363323129985824,
    "right_joint4": A7_JOINT4_FORWARD_RAD,
}
A7_ARM_ACTUATOR_KP = 180.0
A7_ARM_ACTUATOR_KV = 28.0
A7_ARM_JOINT_DAMPING = 2.0
A7_ARM_JOINT_ARMATURE = 0.03
A7_ARM_JOINT_FRICTIONLOSS = 0.02
HAND_CONTROL_JOINTS = [
    "thunb_cmc_roll",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]
PI = "3.141592653589793"
HAND_MOUNT_RPY = f"{PI} 0 {PI}"


def _urdf_with_meshdir(urdf_path: Path) -> Path:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mj_tag = ET.Element("mujoco")
    compiler = ET.SubElement(mj_tag, "compiler")
    compiler.set("meshdir", str((urdf_path.parent / "meshes").resolve()))
    compiler.set("strippath", "true")
    compiler.set("balanceinertia", "true")
    compiler.set("discardvisual", "false")
    root.insert(0, mj_tag)

    out_dir = Path(tempfile.mkdtemp())
    out = out_dir / urdf_path.name
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def _patch_hand_mjcf(mjcf_path: Path, urdf_path: Path) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str((urdf_path.parent / "meshes").resolve()))
    compiler.set("autolimits", "true")

    urdf_root = ET.parse(urdf_path).getroot()
    mimic_pairs = []
    for joint in urdf_root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        mimic_pairs.append(
            (
                joint.attrib["name"],
                mimic.attrib["joint"],
                float(mimic.attrib.get("multiplier", 1.0)),
                float(mimic.attrib.get("offset", 0.0)),
            )
        )

    if mimic_pairs:
        eq_root = root.find("equality")
        if eq_root is None:
            eq_root = ET.SubElement(root, "equality")
        for follower, leader, multiplier, offset in mimic_pairs:
            ET.SubElement(
                eq_root,
                "joint",
                {
                    "joint1": follower,
                    "joint2": leader,
                    "polycoef": f"{offset} {multiplier} 0 0 0",
                    "solimp": "0.95 0.99 0.001 0.5 2",
                    "solref": "0.005 1",
                },
            )

    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)


def _urdf_to_hand_spec(urdf_path: Path) -> mujoco.MjSpec:
    patched_urdf = _urdf_with_meshdir(urdf_path)
    model = mujoco.MjModel.from_xml_path(str(patched_urdf))
    tmp = patched_urdf.with_suffix(".xml")
    mujoco.mj_saveLastXML(str(tmp), model)
    _patch_hand_mjcf(tmp, urdf_path)
    return mujoco.MjSpec.from_file(str(tmp))


def _set_urdf_meshes_absolute(root: ET.Element, mesh_dir: Path) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh.attrib["filename"] = str((mesh_dir / Path(filename).name).resolve())


def _prefix_urdf_tree(root: ET.Element, prefix: str, mesh_dir: Path) -> list[ET.Element]:
    names: dict[str, str] = {}
    for link in root.findall("link"):
        names[link.attrib["name"]] = f"{prefix}{link.attrib['name']}"
    for joint in root.findall("joint"):
        names[joint.attrib["name"]] = f"{prefix}{joint.attrib['name']}"

    items = []
    for child in root:
        if child.tag not in {"link", "joint"}:
            continue
        copied = ET.fromstring(ET.tostring(child, encoding="unicode"))
        for elem in copied.iter():
            if elem.tag in {"link", "joint"} and "name" in elem.attrib:
                elem.attrib["name"] = names.get(elem.attrib["name"], elem.attrib["name"])
            if elem.tag in {"parent", "child"} and "link" in elem.attrib:
                elem.attrib["link"] = names.get(elem.attrib["link"], elem.attrib["link"])
            if elem.tag == "mimic" and "joint" in elem.attrib:
                elem.attrib["joint"] = names.get(elem.attrib["joint"], elem.attrib["joint"])
        _set_urdf_meshes_absolute(copied, mesh_dir)
        items.append(copied)
    return items


def _write_combined_urdf() -> Path:
    a7_root = ET.parse(A7BLACK_DUAL_URDF).getroot()
    a7_root.attrib["name"] = "dual_a7black_realhand_l6"
    _set_urdf_meshes_absolute(a7_root, A7BLACK_DUAL_URDF.parent / "meshes")

    mj_tag = ET.Element("mujoco")
    compiler = ET.SubElement(mj_tag, "compiler")
    compiler.set("strippath", "false")
    compiler.set("balanceinertia", "true")
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    a7_root.insert(0, mj_tag)

    hand_specs = {
        "left": ("lh_", "left_link8", HAND_URDFS["left"]),
        "right": ("rh_", "right_link8", HAND_URDFS["right"]),
    }
    for side, (prefix, parent_link, hand_urdf) in hand_specs.items():
        hand_root = ET.parse(hand_urdf).getroot()
        for item in _prefix_urdf_tree(hand_root, prefix, hand_urdf.parent / "meshes"):
            a7_root.append(item)

        fixed = ET.SubElement(a7_root, "joint", {"name": f"{side}_hand_mount", "type": "fixed"})
        ET.SubElement(fixed, "origin", {"xyz": "0 0 0", "rpy": HAND_MOUNT_RPY})
        ET.SubElement(fixed, "parent", {"link": parent_link})
        ET.SubElement(fixed, "child", {"link": f"{prefix}hand_base_link"})

    out_dir = Path(tempfile.mkdtemp())
    out = out_dir / "dual_a7black_realhand_l6.urdf"
    ET.indent(a7_root, space="  ")
    ET.ElementTree(a7_root).write(out, encoding="utf-8", xml_declaration=True)
    return out


def _add_mimic_equalities(spec: mujoco.MjSpec, urdf_path: Path) -> None:
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = joint.attrib["name"]
        eq.name2 = mimic.attrib["joint"]
        eq.data = [
            float(mimic.attrib.get("offset", 0.0)),
            float(mimic.attrib.get("multiplier", 1.0)),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        eq.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
        eq.solref = [0.005, 1.0]


def _add_position_actuator(
    spec: mujoco.MjSpec,
    joint_name: str,
    kp: float,
    kv: float,
    ctrlrange: tuple[float, float] | None = None,
    forcerange: tuple[float, float] | None = None,
) -> None:
    act = spec.add_actuator()
    act.name = f"{joint_name}_pos"
    act.target = joint_name
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.set_to_position(kp=kp, kv=kv)
    if ctrlrange is not None:
        act.ctrlrange = ctrlrange
        act.ctrllimited = 1
    if forcerange is not None:
        act.forcerange = forcerange
        act.forcelimited = 1


def _add_a7_arm_actuators(spec: mujoco.MjSpec) -> None:
    for joint_name in A7_ARM_JOINTS:
        if spec.joint(joint_name) is None:
            raise RuntimeError(f"A7black joint not found in MuJoCo spec: {joint_name}")
        _add_position_actuator(
            spec,
            joint_name,
            kp=A7_ARM_ACTUATOR_KP,
            kv=A7_ARM_ACTUATOR_KV,
        )


def _add_hand_actuators(spec: mujoco.MjSpec, prefix: str) -> None:
    for joint_name in HAND_CONTROL_JOINTS:
        full_name = f"{prefix}{joint_name}"
        if spec.joint(full_name) is None:
            continue
        _add_position_actuator(
            spec,
            full_name,
            kp=20.0,
            kv=1.0,
            ctrlrange=(0.0, 1.4),
            forcerange=(-5.0, 5.0),
        )


def _absolutize_mesh_paths(spec: mujoco.MjSpec, prefix: str, meshes_dir: Path) -> None:
    abs_dir = str(meshes_dir.resolve())
    for mesh in spec.meshes:
        if not mesh.name.startswith(prefix):
            continue
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.join(abs_dir, os.path.basename(mesh.file))


def _add_attachment_site(parent: mujoco.MjSpec, body_name: str, site_name: str) -> mujoco.MjsSite:
    body = parent.body(body_name)
    if body is None:
        raise RuntimeError(f"A7black body not found for hand attachment: {body_name}")
    site = body.add_site()
    site.name = site_name
    site.pos = [0.0, 0.0, 0.0]
    site.quat = [0.0, 0.0, 0.0, 1.0]
    site.size = [0.01, 0.0, 0.0]
    site.rgba = [0.1, 0.7, 1.0, 0.35]
    return site


def _add_mocap_target(parent: mujoco.MjSpec, name: str, pos: list[float]) -> None:
    body = parent.worldbody.add_body()
    body.name = name
    body.mocap = True
    body.pos = pos
    geom = body.add_geom()
    geom.name = f"{name}_marker"
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [0.025, 0.0, 0.0]
    geom.rgba = [0.0, 0.75, 1.0, 0.45]
    geom.contype = 0
    geom.conaffinity = 0


def _add_scene(parent: mujoco.MjSpec) -> None:
    light = parent.worldbody.add_light()
    light.name = "top_light"
    light.pos = [0.0, 0.0, 2.5]
    light.dir = [0.0, 0.0, -1.0]
    light.diffuse = [0.8, 0.85, 1.0]
    light.ambient = [0.25, 0.3, 0.4]

    floor = parent.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [1.0, 1.0, 0.01]
    floor.pos = [0.0, 0.0, -0.01]
    floor.rgba = [0.1, 0.15, 0.2, 1.0]
    floor.contype = 1
    floor.conaffinity = 1


def _set_initial_arm_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_name, qpos in A7_READY_JOINTS_RAD.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"A7black joint not found in MuJoCo model: {joint_name}")
        data.qpos[model.jnt_qposadr[jid]] = qpos

        actid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_pos")
        if actid >= 0:
            data.ctrl[actid] = qpos


def build_combined_spec() -> mujoco.MjSpec:
    if not A7BLACK_DUAL_URDF.exists():
        raise FileNotFoundError(f"Missing dual A7black URDF: {A7BLACK_DUAL_URDF}")

    combined_urdf = _write_combined_urdf()
    parent = mujoco.MjSpec.from_file(str(combined_urdf))
    _add_mimic_equalities(parent, combined_urdf)
    _add_a7_arm_actuators(parent)
    _add_hand_actuators(parent, "lh_")
    _add_hand_actuators(parent, "rh_")

    _add_mocap_target(parent, "left_target", [0.0, 0.41, 0.35])
    _add_mocap_target(parent, "right_target", [0.0, -0.41, 0.35])
    _add_scene(parent)
    parent.meshdir = str((A7BLACK_DUAL_URDF.parent / "meshes").resolve())
    return parent


def write_combined_xml(out_path: Path) -> Path:
    spec = build_combined_spec()
    spec.compile()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_xml_with_absolute_hand_meshes(spec))
    return out_path


def _xml_with_absolute_hand_meshes(spec: mujoco.MjSpec) -> str:
    """Saveable MJCF with L6 hand meshes pinned to their source folders."""
    root = ET.fromstring(spec.to_xml())
    _disable_gravity(root)
    _stabilize_a7_arm_joints(root)
    _apply_ur5_grid_floor(root)

    asset = root.find("asset")
    if asset is None:
        return spec.to_xml()

    mesh_dirs = {
        "lh_": HAND_URDFS["left"].parent / "meshes",
        "rh_": HAND_URDFS["right"].parent / "meshes",
    }
    for mesh in asset.findall("mesh"):
        name = mesh.attrib.get("name", "")
        for prefix, mesh_dir in mesh_dirs.items():
            if name.startswith(prefix) and "file" in mesh.attrib:
                mesh.attrib["file"] = str((mesh_dir / Path(mesh.attrib["file"]).name).resolve())
                break

    ET.indent(root, space="  ")
    buf = StringIO()
    ET.ElementTree(root).write(buf, encoding="unicode", xml_declaration=True)
    return buf.getvalue()


def _disable_gravity(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("gravity", "0 0 0")


def _stabilize_a7_arm_joints(root: ET.Element) -> None:
    for joint_name in A7_ARM_JOINTS:
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            continue
        joint.set("damping", str(A7_ARM_JOINT_DAMPING))
        joint.set("armature", str(A7_ARM_JOINT_ARMATURE))
        joint.set("frictionloss", str(A7_ARM_JOINT_FRICTIONLOSS))


def _apply_ur5_grid_floor(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody = root.find("worldbody")
        insert_idx = list(root).index(worldbody) if worldbody is not None else 0
        root.insert(insert_idx, asset)

    if asset.find("./texture[@name='grid']") is None:
        asset.insert(
            0,
            ET.Element(
                "texture",
                {
                    "name": "grid",
                    "type": "2d",
                    "builtin": "checker",
                    "rgb1": ".2 .3 .4",
                    "rgb2": ".1 0.15 0.2",
                    "width": "512",
                    "height": "512",
                    "mark": "cross",
                    "markrgb": ".8 .8 .8",
                },
            ),
        )

    if asset.find("./material[@name='grid']") is None:
        asset.insert(
            1,
            ET.Element(
                "material",
                {
                    "name": "grid",
                    "texture": "grid",
                    "texrepeat": "1 1",
                    "texuniform": "true",
                },
            ),
        )

    floor = root.find("./worldbody/geom[@name='floor']")
    if floor is not None:
        floor.set("size", "1 1 0.01")
        floor.set("material", "grid")
        floor.attrib.pop("rgba", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-xml", default="", help="Path to save the generated MJCF.")
    parser.add_argument("--no-viewer", action="store_true", help="Compile only; skip the viewer.")
    args = parser.parse_args()

    spec = build_combined_spec()
    xml = _xml_with_absolute_hand_meshes(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    _set_initial_arm_pose(model, data)
    mujoco.mj_forward(model, data)

    if args.save_xml:
        out = Path(args.save_xml).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(xml)
        print(f"Saved combined MJCF -> {out}")

    print(
        f"Compiled model: nq={model.nq}, nv={model.nv}, "
        f"nbody={model.nbody}, neq={model.neq}, nu={model.nu}"
    )

    if args.no_viewer:
        return

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.45]
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
