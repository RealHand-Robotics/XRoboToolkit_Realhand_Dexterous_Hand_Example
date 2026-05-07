"""MuJoCo visualization for the RealHand LZ01_Lite humanoid.

The model is loaded from:
  assets/real_hand/LZ01_Lite/mjcf/mjmodel.xml

Gravity is enabled by default, so the free-base robot will move under physics
unless you add balancing/control later.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import mujoco
from mujoco import viewer as mj_viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets" / "real_hand" / "LZ01_Lite"
LZ01_LITE_MJCF = ASSET_DIR / "mjcf" / "mjmodel.xml"
LZ01_LITE_MESH_DIR = ASSET_DIR / "meshes"


def _patched_xml(mjcf_path: Path, gravity: str) -> str:
    root = ET.parse(mjcf_path).getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(LZ01_LITE_MESH_DIR.resolve()))

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("gravity", gravity)

    ground = root.find("./worldbody/geom[@name='ground']")
    if ground is not None:
        ground.set("size", "2 2 0.01")

    ET.indent(root, space="  ")
    buf = StringIO()
    ET.ElementTree(root).write(buf, encoding="unicode", xml_declaration=True)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf-path", type=Path, default=LZ01_LITE_MJCF)
    parser.add_argument("--gravity", default="0 0 -9.81", help='MuJoCo gravity vector, e.g. "0 0 -9.81".')
    parser.add_argument("--save-xml", type=Path, default=None, help="Optional path to save the patched MJCF.")
    parser.add_argument("--no-viewer", action="store_true", help="Compile only; skip the viewer.")
    args = parser.parse_args()

    if not args.mjcf_path.exists():
        raise FileNotFoundError(f"Missing LZ01_Lite MJCF: {args.mjcf_path}")
    if not LZ01_LITE_MESH_DIR.exists():
        raise FileNotFoundError(f"Missing LZ01_Lite mesh directory: {LZ01_LITE_MESH_DIR}")

    xml = _patched_xml(args.mjcf_path, args.gravity)
    if args.save_xml is not None:
        args.save_xml.parent.mkdir(parents=True, exist_ok=True)
        args.save_xml.write_text(xml)
        print(f"Saved patched LZ01_Lite MJCF -> {args.save_xml.resolve()}")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(
        f"Compiled LZ01_Lite: nq={model.nq}, nv={model.nv}, "
        f"nbody={model.nbody}, nu={model.nu}, gravity={model.opt.gravity}"
    )

    if args.no_viewer:
        return

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
