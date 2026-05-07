"""MuJoCo simulation for dual UR5e arms with RealHand L6 dexterous hands.

Combines the existing dual UR5e MJCF (assets/universal_robots_ur5e/scene_dual_arm.xml)
with the left/right RealHand L6 URDFs (assets/real_hand/l6/{left,right}/...) by
attaching each hand to its arm's wrist attachment site at runtime via mujoco.MjSpec.

Each URDF is first converted to MJCF (via MjModel.from_xml_path + mj_saveLastXML),
then patched with:
  - a <compiler meshdir=...> pointing at the hand's mesh directory,
  - <equality> joint constraints that emulate the URDF mimic tags
    (the dip joints follow the proximal pitch joints).

Position actuators are added for the six independent control joints per hand
(thumb roll, thumb pitch, plus index/middle/ring/pinky MCP pitches). The dip
joints follow via the equality constraints.

Run:
    python scripts/simulation/sim_dual_ur5e_realhand_l6_mujoco.py

Optional flags (see --help):
    --save-xml PATH   Save the combined compiled MJCF to PATH for inspection.
    --no-viewer       Skip the viewer (useful for headless validation).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from mujoco import viewer as mj_viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"

UR5E_SCENE = ASSETS / "universal_robots_ur5e" / "scene_dual_arm.xml"
HAND_URDFS = {
    "left": ASSETS / "real_hand" / "l6" / "left" / "realhand_l6_left.urdf",
    "right": ASSETS / "real_hand" / "l6" / "right" / "realhand_l6_right.urdf",
}

HAND_CONTROL_JOINTS = [
    "thunb_cmc_roll",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


def _patch_hand_mjcf(mjcf_path: Path, urdf_path: Path) -> None:
    """Patch the auto-converted MJCF in place.

    - Sets meshdir so the mesh basenames resolve to the URDF's meshes/ folder.
    - Adds <equality> joint coupling for every <mimic> tag in the URDF
      (MuJoCo's URDF importer drops these silently).
    """
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


def _urdf_with_meshdir(urdf_path: Path) -> Path:
    """Copy the URDF to a temp file and inject <mujoco><compiler meshdir=...>.

    MuJoCo's URDF parser strips the directory prefix from mesh filenames and
    looks for the basenames in compiler.meshdir (or the URDF's own directory).
    Pointing meshdir at the absolute meshes/ folder makes the load succeed.
    """
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


def _urdf_to_hand_spec(urdf_path: Path) -> mujoco.MjSpec:
    """Convert a RealHand L6 URDF to an MjSpec with mimic equalities patched in."""
    patched_urdf = _urdf_with_meshdir(urdf_path)
    model = mujoco.MjModel.from_xml_path(str(patched_urdf))
    tmp = patched_urdf.with_suffix(".xml")
    mujoco.mj_saveLastXML(str(tmp), model)
    _patch_hand_mjcf(tmp, urdf_path)
    return mujoco.MjSpec.from_file(str(tmp))


def _add_hand_actuators(spec: mujoco.MjSpec, prefix: str) -> None:
    """Position actuators for the six driven joints of one hand."""
    for jname in HAND_CONTROL_JOINTS:
        full_name = f"{prefix}{jname}"
        if spec.joint(full_name) is None:
            continue
        act = spec.add_actuator()
        act.name = f"{prefix}{jname}_pos"
        act.target = full_name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=20.0, kv=1.0)
        act.ctrlrange = (0.0, 1.4)
        act.ctrllimited = 1
        act.forcerange = (-5.0, 5.0)
        act.forcelimited = 1


def _extend_keyframe_ctrl(parent: mujoco.MjSpec, n_arm_actuators: int = 12) -> None:
    """``MjSpec.attach`` extends each keyframe's qpos but not its ctrl.

    Append zeros for the new hand actuators so a viewer / controller can
    safely call ``mj_resetDataKeyframe``.
    """
    target_ctrl_len = n_arm_actuators + 2 * len(HAND_CONTROL_JOINTS)
    for key in parent.keys:
        ctrl = list(key.ctrl)
        if len(ctrl) < target_ctrl_len:
            ctrl += [0.0] * (target_ctrl_len - len(ctrl))
        key.ctrl = ctrl


def _quat_mul(q1, q2):
    """Hamilton product of two [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


# 180-degree rotation around the local Z axis as a [w, x, y, z] quaternion.
# Composed onto each attachment-site quat below to flip both hands at the wrist.
ROT_Z_180_QUAT = [0.0, 0.0, 0.0, 1.0]


def _absolutize_mesh_paths(parent: mujoco.MjSpec, prefix: str, meshes_dir: Path) -> None:
    """Rewrite hand mesh ``file`` attributes to absolute paths.

    The parent spec has a single ``meshdir`` (set to the UR5e asset folder),
    so mesh basenames inherited from the attached hand spec would otherwise
    look in the wrong place.
    """
    abs_dir = str(meshes_dir.resolve())
    for mesh in parent.meshes:
        if not mesh.name.startswith(prefix):
            continue
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.join(abs_dir, os.path.basename(mesh.file))


def build_combined_spec() -> mujoco.MjSpec:
    parent = mujoco.MjSpec.from_file(str(UR5E_SCENE))

    for side, urdf_path in HAND_URDFS.items():
        if not urdf_path.exists():
            raise FileNotFoundError(f"Missing RealHand L6 URDF: {urdf_path}")
        hand_spec = _urdf_to_hand_spec(urdf_path)
        site_name = f"{side}_attachment_site"
        site = parent.site(site_name)
        if site is None:
            raise RuntimeError(f"UR5e scene is missing site '{site_name}'")
        prefix = f"{side[0]}h_"
        site.quat = _quat_mul(list(site.quat), ROT_Z_180_QUAT)
        parent.attach(hand_spec, prefix=prefix, site=site)
        _add_hand_actuators(parent, prefix)
        _absolutize_mesh_paths(parent, prefix, urdf_path.parent / "meshes")

    _extend_keyframe_ctrl(parent)
    return parent


def write_combined_xml(out_path: Path) -> Path:
    """Build the combined model and save it to ``out_path``. Returns the resolved path.

    Re-points the parent's ``meshdir`` (originally a relative ``"assets"`` from
    the UR5e MJCF) at the absolute UR5e asset directory so the saved file can
    be loaded from any location.
    """
    spec = build_combined_spec()
    spec.meshdir = str((UR5E_SCENE.parent / "assets").resolve())
    spec.compile()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(spec.to_xml())
    return out_path


def _set_initial_arm_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Place arms in a comfortable away-from-singularity pose."""
    home = {
        "left_shoulder_pan_joint": 1.57,
        "left_shoulder_lift_joint": -1.57,
        "left_elbow_joint": 1.57,
        "left_wrist_1_joint": -1.57,
        "left_wrist_2_joint": -1.57,
        "left_wrist_3_joint": 0.0,
        "right_shoulder_pan_joint": -1.57,
        "right_shoulder_lift_joint": -1.57,
        "right_elbow_joint": -1.57,
        "right_wrist_1_joint": -1.57,
        "right_wrist_2_joint": 1.57,
        "right_wrist_3_joint": 0.0,
    }
    for jname, qpos in home.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = qpos
        actid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if actid >= 0:
            data.ctrl[actid] = qpos
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-xml", default="", help="Path to save the compiled combined MJCF.")
    parser.add_argument("--no-viewer", action="store_true", help="Compile only; skip the viewer.")
    args = parser.parse_args()

    spec = build_combined_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    _set_initial_arm_pose(model, data)

    if args.save_xml:
        out = Path(args.save_xml).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(spec.to_xml())
        print(f"Saved combined MJCF -> {out}")

    print(
        f"Compiled model: nq={model.nq}, nv={model.nv}, "
        f"nbody={model.nbody}, neq={model.neq}, nu={model.nu}"
    )

    if args.no_viewer:
        return

    with mj_viewer.launch_passive(model, data) as v:
        v.cam.azimuth = 130
        v.cam.elevation = -20
        v.cam.distance = 1.6
        v.cam.lookat[:] = [0.0, 0.0, 0.5]
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
