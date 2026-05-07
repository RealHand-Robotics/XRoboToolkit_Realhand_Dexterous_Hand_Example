"""Teleop controllers for the RealHand L6 dexterous hand.

Uses the new ``realhand`` Python SDK (``pip install
git+https://github.com/RealHand-Robotics/realbot-python-sdk.git``) and
``dex_retargeting`` to drive one or two L6 hands from XR controller / XR
hand-tracking input.
"""

import os
import threading
import time
from typing import List, Optional

import numpy as np
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from realhand import L6

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.realhand_hardware_map import (
    DEFAULT_REALHAND_CAN_PORTS,
)
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_LEFT_URDF = os.path.join(ASSET_PATH, "real_hand/l6/left")
DEFAULT_RIGHT_URDF = os.path.join(ASSET_PATH, "real_hand/l6/right")
DEFAULT_LEFT_HAND_CAN_PORT = DEFAULT_REALHAND_CAN_PORTS["left_hand"]
DEFAULT_RIGHT_HAND_CAN_PORT = DEFAULT_REALHAND_CAN_PORTS["right_hand"]

L6_OPEN_ANGLE = 100.0
L6_CLOSED_ANGLE = 0.0
L6_RETARGET_MAX_RAD = np.pi / 2.0

# Angle range of each finger in the L6 SDK (0 = fully closed, 100 = fully open).
DEFAULT_OPEN_ANGLES: List[float] = [L6_OPEN_ANGLE] * 6
DEFAULT_CLOSED_ANGLES: List[float] = [
    L6_CLOSED_ANGLE,
    L6_OPEN_ANGLE,
    L6_CLOSED_ANGLE,
    L6_CLOSED_ANGLE,
    L6_CLOSED_ANGLE,
    L6_CLOSED_ANGLE,
]

CONTROLLER_DEADZONE = 0.05


def _qpos_to_l6_angles(qpos: np.ndarray) -> Optional[List[float]]:
    """Convert the 11-dof vector from dex_retargeting to the 6 joint
    angles expected by the L6 SDK, scaled into the 0-100 range.

    The retargeting returns flexion values roughly in ``[0, pi/2]`` where
    larger means more closed. The L6 SDK uses ``100`` for open and ``0`` for
    closed, so the normalized flexion value is inverted before sending.
    """
    if qpos is None or len(qpos) != 11:
        return None

    rad = np.array(
        [
            (qpos[9] + qpos[10]) / 2.0,  # pinky
            qpos[8],                     # thumb abd
            qpos[0],                     # index
            qpos[3],                     # middle
            qpos[5],                     # ring
            qpos[6],                     # thumb flex
        ],
        dtype=float,
    )
    normalized_flexion = np.clip(rad / L6_RETARGET_MAX_RAD, 0.0, 1.0)
    return ((1.0 - normalized_flexion) * L6_OPEN_ANGLE).tolist()


class RealHandL6Controller:
    """Single-hand teleop controller for a RealHand L6.

    Parameters
    ----------
    xr_client:
        Shared :class:`XrClient` used to read XR tracking / trigger state.
    hand_type:
        ``"left"`` or ``"right"``. Selects the URDF, the dex_retargeting hand
        type, and the arbitration ID inside the L6 SDK.
    interface_name:
        Name of the SocketCAN interface attached to the hand. If omitted,
        left defaults to ``can2`` and right defaults to ``can3``.
    urdf_path:
        Optional override for the L6 URDF path.
    control_hz:
        Target update rate for the retarget + command loop.
    """

    def __init__(
        self,
        xr_client: XrClient,
        hand_type: str = "right",
        interface_name: Optional[str] = None,
        urdf_path: Optional[str] = None,
        control_hz: float = 60.0,
    ) -> None:
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got {hand_type!r}")

        self.xr_client = xr_client
        self.hand_type = hand_type
        self.interface_name = interface_name or (
            DEFAULT_LEFT_HAND_CAN_PORT
            if hand_type == "left"
            else DEFAULT_RIGHT_HAND_CAN_PORT
        )
        self.control_hz = control_hz
        self._period = 1.0 / max(control_hz, 1.0)

        urdf_path = urdf_path or (
            DEFAULT_LEFT_URDF if hand_type == "left" else DEFAULT_RIGHT_URDF
        )
        self.urdf_path = urdf_path

        self.tracker = DexHandTracker(
            robot_name=RobotName.real,
            urdf_path=urdf_path,
            hand_type=HandType.left if hand_type == "left" else HandType.right,
            retargeting_type=RetargetingType.vector,
        )

        # New RealHand SDK (realbot-python-sdk)
        self.hand = L6(side=hand_type, interface_name=self.interface_name)
        self.hand.angle.set_angles(DEFAULT_OPEN_ANGLES)

        self._target_angles: List[float] = list(DEFAULT_OPEN_ANGLES)
        self._target_lock = threading.Lock()

    # ------------------------------------------------------------------ retarget
    def _compute_target(self) -> List[float]:
        hand_state = self.xr_client.get_hand_tracking_state(self.hand_type)
        trigger_val = self.xr_client.get_key_value_by_name(f"{self.hand_type}_trigger")

        if hand_state is not None:
            hand_state = np.array(hand_state)
            if not np.all(hand_state == 0):
                mediapipe = pico_hand_state_to_mediapipe(hand_state)
                qpos = self.tracker.retarget(mediapipe)
                angles = _qpos_to_l6_angles(qpos)
                if angles is not None:
                    return angles

        # Fallback: linearly interpolate open <-> closed with the trigger value.
        if trigger_val < CONTROLLER_DEADZONE:
            return list(DEFAULT_OPEN_ANGLES)
        t = float(np.clip(trigger_val, 0.0, 1.0))
        return [
            (1.0 - t) * o + t * c
            for o, c in zip(DEFAULT_OPEN_ANGLES, DEFAULT_CLOSED_ANGLES)
        ]

    # ------------------------------------------------------------------ threads
    def run_retarget_thread(self, stop_event: threading.Event) -> None:
        print(f"[RealHand {self.hand_type}] retarget thread started")
        while not stop_event.is_set():
            start = time.time()
            try:
                target = self._compute_target()
                with self._target_lock:
                    self._target_angles = target
            except Exception as exc:
                print(f"[RealHand {self.hand_type}] retarget error: {exc}")
            elapsed = time.time() - start
            time.sleep(max(0.0, self._period - elapsed))

    def run_control_thread(self, stop_event: threading.Event) -> None:
        print(f"[RealHand {self.hand_type}] control thread started")
        while not stop_event.is_set():
            start = time.time()
            with self._target_lock:
                target = list(self._target_angles)
            target = np.clip(target, L6_CLOSED_ANGLE, L6_OPEN_ANGLE).tolist()
            try:
                self.hand.angle.set_angles(target)
            except Exception as exc:
                print(f"[RealHand {self.hand_type}] set_angles error: {exc}")
            elapsed = time.time() - start
            time.sleep(max(0.0, self._period - elapsed))

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Send a fully-open command to the hand."""
        self.hand.angle.set_angles(DEFAULT_OPEN_ANGLES)

    def close(self) -> None:
        hand = getattr(self, "hand", None)
        if hand is None:
            return
        try:
            hand.close()
        except Exception as exc:
            print(f"[RealHand {self.hand_type}] close error: {exc}")

    def __del__(self) -> None:
        self.close()


class DualRealHandL6Controller:
    """Convenience wrapper that manages a left and a right L6 hand together."""

    def __init__(
        self,
        xr_client: XrClient,
        left_interface: str = DEFAULT_LEFT_HAND_CAN_PORT,
        right_interface: str = DEFAULT_RIGHT_HAND_CAN_PORT,
        left_urdf: Optional[str] = None,
        right_urdf: Optional[str] = None,
        control_hz: float = 60.0,
    ) -> None:
        self.left = RealHandL6Controller(
            xr_client,
            hand_type="left",
            interface_name=left_interface,
            urdf_path=left_urdf,
            control_hz=control_hz,
        )
        self.right = RealHandL6Controller(
            xr_client,
            hand_type="right",
            interface_name=right_interface,
            urdf_path=right_urdf,
            control_hz=control_hz,
        )
        self._threads: List[threading.Thread] = []
        self._stop_event: Optional[threading.Event] = None

    def start_threads(
        self,
        stop_event: Optional[threading.Event] = None,
        daemon: bool = True,
    ) -> List[threading.Thread]:
        """Start left/right retarget and control threads."""
        if self._threads:
            raise RuntimeError("DualRealHandL6Controller threads are already running")

        self._stop_event = stop_event or threading.Event()
        self._threads = [
            threading.Thread(
                name="left_l6_retarget",
                target=self.left.run_retarget_thread,
                args=(self._stop_event,),
                daemon=daemon,
            ),
            threading.Thread(
                name="left_l6_control",
                target=self.left.run_control_thread,
                args=(self._stop_event,),
                daemon=daemon,
            ),
            threading.Thread(
                name="right_l6_retarget",
                target=self.right.run_retarget_thread,
                args=(self._stop_event,),
                daemon=daemon,
            ),
            threading.Thread(
                name="right_l6_control",
                target=self.right.run_control_thread,
                args=(self._stop_event,),
                daemon=daemon,
            ),
        ]

        for thread in self._threads:
            thread.start()
        return list(self._threads)

    def stop_threads(self, join_timeout: float = 2.0) -> None:
        """Stop hand threads and wait briefly for them to exit."""
        if self._stop_event is not None:
            self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=join_timeout)

        self._threads = []
        self._stop_event = None

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()

    def close(self) -> None:
        self.stop_threads()
        self.left.close()
        self.right.close()
