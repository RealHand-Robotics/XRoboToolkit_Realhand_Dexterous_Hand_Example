import time
from typing import List, Literal, Optional, Tuple, Union

import meshcat.transformations as tf
import numpy as np
from realhand import A7, Pose


class A7BlackInterface:
    """Hardware interface for one RealHand A7black arm.

    The RealHand SDK exposes A7 arm control through the ``A7`` class.  This
    wrapper keeps the same shape as the existing hardware interfaces in this
    repository while leaving the CAN interface and arm side configurable.
    """

    NUM_JOINTS = 7

    def __init__(
        self,
        can_port: str = "can0",
        side: Literal["left", "right"] = "left",
        interface_type: str = "socketcan",
        world_frame: Literal["urdf", "maestro"] = "urdf",
        tcp_offset: Optional[List[float]] = None,
        default_velocity: float = 1.0,
        default_acceleration: float = 10.0,
        enable_on_start: bool = True,
        check_joint_limits: bool = True,
        use_streaming_setpoint: bool = True,
        joint2_pole_clearance_deg: float = 10.0,
        home_timeout_sec: float = 20.0,
        home_tolerance_deg: float = 2.0,
    ):
        self.can_port = can_port
        self.side = side
        self.interface_type = interface_type
        self.world_frame = world_frame
        self.tcp_offset = list(tcp_offset) if tcp_offset is not None else [0.0, 0.0, 0.0]
        self.check_joint_limits = check_joint_limits
        self.use_streaming_setpoint = use_streaming_setpoint
        self.joint2_pole_clearance_rad = np.deg2rad(float(joint2_pole_clearance_deg))
        self.home_timeout_sec = float(home_timeout_sec)
        self.home_tolerance_rad = np.deg2rad(float(home_tolerance_deg))
        self.default_velocity = float(default_velocity)
        self.default_acceleration = float(default_acceleration)
        self._closed = False

        self.arm = A7(
            side=side,
            interface_name=can_port,
            interface_type=interface_type,
            tcp_offset=self.tcp_offset,
            world_frame=world_frame,
        )
        self._joint_limits = list(self.arm.get_joint_limits())
        self._apply_joint2_pole_clearance()

        if enable_on_start:
            self.enable()

        self.set_motion_limits(default_velocity, default_acceleration)

    def get_joint_names(self) -> List[str]:
        return [f"{self.side}_joint{i}" for i in range(1, self.NUM_JOINTS + 1)]

    @staticmethod
    def urdf_to_sdk_joint_positions(positions: Union[List[float], np.ndarray]) -> List[float]:
        """Convert repo A7black URDF joint order to RealHand SDK joint order.

        The repo URDF uses joint6=wrist roll and joint7=wrist pitch. The A7 SDK
        uses joint6=wrist pitch and joint7=wrist roll. The pitch axis sign is
        opposite between the repo URDF and the SDK joint definition.
        """
        q = np.asarray(positions, dtype=float).reshape(-1)
        if q.shape[0] != A7BlackInterface.NUM_JOINTS:
            raise ValueError(
                f"A7black expects {A7BlackInterface.NUM_JOINTS} joints, got {q.shape[0]}"
            )
        return [q[0], q[1], q[2], q[3], q[4], -q[6], q[5]]

    @staticmethod
    def sdk_to_urdf_joint_positions(positions: Union[List[float], np.ndarray]) -> List[float]:
        """Convert RealHand SDK joint order to repo A7black URDF joint order."""
        q = np.asarray(positions, dtype=float).reshape(-1)
        if q.shape[0] != A7BlackInterface.NUM_JOINTS:
            raise ValueError(
                f"A7black expects {A7BlackInterface.NUM_JOINTS} joints, got {q.shape[0]}"
            )
        return [q[0], q[1], q[2], q[3], q[4], q[6], -q[5]]

    def enable(self) -> bool:
        self.arm.enable()
        return True

    def disable(self) -> bool:
        self.arm.disable()
        return True

    def reset_error(self) -> bool:
        self.arm.reset_error()
        return True

    def go_home(self, blocking: bool = True) -> bool:
        self.arm.enable()
        q_home = self.clip_joint_positions([0.0] * self.NUM_JOINTS)
        print(f"[A7black {self.side}] moving to safe home (deg): {np.rad2deg(q_home)}")
        self.arm.move_j(q_home, blocking=False)
        if not blocking:
            return True
        return self.wait_until_joint_positions(
            q_home,
            timeout_sec=self.home_timeout_sec,
            tolerance_rad=self.home_tolerance_rad,
        )

    def wait_until_joint_positions(
        self,
        positions: Union[List[float], np.ndarray],
        timeout_sec: float,
        tolerance_rad: float,
        poll_period_sec: float = 0.05,
    ) -> bool:
        q_target = np.asarray(positions, dtype=float).reshape(-1)
        deadline = time.time() + timeout_sec
        last_error = np.inf

        while time.time() < deadline:
            q_current = np.asarray(self.get_joint_positions(), dtype=float)
            last_error = float(np.max(np.abs(q_current - q_target)))
            if last_error <= tolerance_rad:
                return True
            time.sleep(poll_period_sec)

        print(
            f"[A7black {self.side}] home timeout: max error "
            f"{np.rad2deg(last_error):.2f} deg"
        )
        return False

    def gravity_compensation(self) -> bool:
        self.disable()
        return True

    def protect_mode(self) -> bool:
        self.disable()
        return True

    def emergency_stop(self) -> bool:
        self.arm.emergency_stop()
        return True

    def set_motion_limits(self, velocity: float, acceleration: float) -> bool:
        self.arm.set_velocities([float(velocity)] * self.NUM_JOINTS)
        self.arm.set_accelerations([float(acceleration)] * self.NUM_JOINTS)
        return True

    def _apply_joint2_pole_clearance(self) -> None:
        if self.joint2_pole_clearance_rad <= 0.0:
            return

        joint2_index = 1
        low, high = self._joint_limits[joint2_index]
        if self.side == "left":
            low = max(low, self.joint2_pole_clearance_rad)
        else:
            high = min(high, -self.joint2_pole_clearance_rad)
        self._joint_limits[joint2_index] = (low, high)
        self.arm.set_joint_limits(self._joint_limits)

    def clip_joint_positions(self, positions: Union[List[float], np.ndarray]) -> List[float]:
        q = np.asarray(positions, dtype=float).reshape(-1)
        if q.shape[0] != self.NUM_JOINTS:
            raise ValueError(f"A7black expects {self.NUM_JOINTS} joint positions, got {q.shape[0]}")

        lows = np.array([limit[0] for limit in self._joint_limits], dtype=float)
        highs = np.array([limit[1] for limit in self._joint_limits], dtype=float)
        return np.clip(q, lows, highs).tolist()

    def set_joint_positions(
        self,
        positions: Union[List[float], np.ndarray],
        blocking: bool = False,
        clip_to_limits: bool = True,
        **kwargs,
    ) -> bool:
        q = np.asarray(positions, dtype=float).reshape(-1)
        if q.shape[0] != self.NUM_JOINTS:
            raise ValueError(f"A7black expects {self.NUM_JOINTS} joint positions, got {q.shape[0]}")
        q_des = self.clip_joint_positions(q) if clip_to_limits else q.tolist()

        if self.use_streaming_setpoint and hasattr(self.arm, "_set_angles"):
            # Public move_j rejects updates while its motion timer is active; teleop needs streaming setpoints.
            self.arm._set_angles(q_des, check_limits=self.check_joint_limits)
        else:
            self.arm.move_j(q_des, blocking=blocking)
        return True

    def set_urdf_joint_positions(
        self,
        positions: Union[List[float], np.ndarray],
        blocking: bool = False,
        clip_to_limits: bool = True,
        **kwargs,
    ) -> bool:
        return self.set_joint_positions(
            self.urdf_to_sdk_joint_positions(positions),
            blocking=blocking,
            clip_to_limits=clip_to_limits,
            **kwargs,
        )

    def set_ee_pose(
        self,
        pos: Optional[Union[List[float], np.ndarray]] = None,
        quat: Optional[Union[List[float], np.ndarray]] = None,
        blocking: bool = False,
        **kwargs,
    ) -> bool:
        if pos is None or quat is None:
            raise ValueError("Both pos and quat are required for set_ee_pose")

        roll, pitch, yaw = tf.euler_from_quaternion(np.asarray(quat, dtype=float))
        self.arm.move_p(
            Pose(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
                rx=float(roll),
                ry=float(pitch),
                rz=float(yaw),
            ),
            blocking=blocking,
        )
        return True

    def set_ee_pose_xyzrpy(
        self,
        xyzrpy: Union[List[float], np.ndarray],
        blocking: bool = False,
        **kwargs,
    ) -> bool:
        xyzrpy = np.asarray(xyzrpy, dtype=float).reshape(-1)
        if xyzrpy.shape[0] != 6:
            raise ValueError(f"xyzrpy must contain 6 values, got {xyzrpy.shape[0]}")
        self.arm.move_p(
            Pose(
                x=float(xyzrpy[0]),
                y=float(xyzrpy[1]),
                z=float(xyzrpy[2]),
                rx=float(xyzrpy[3]),
                ry=float(xyzrpy[4]),
                rz=float(xyzrpy[5]),
            ),
            blocking=blocking,
        )
        return True

    def get_joint_positions(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        positions = self.arm.get_angles()
        return self._select_joints(positions, joint_names)

    def get_urdf_joint_positions(self) -> List[float]:
        return self.sdk_to_urdf_joint_positions(self.arm.get_angles())

    def get_joint_velocities(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        velocities = self.arm.get_velocities()
        return self._select_joints(velocities, joint_names)

    def get_urdf_joint_velocities(self) -> List[float]:
        return self.sdk_to_urdf_joint_positions(self.arm.get_velocities())

    def get_joint_currents(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        torques = self.arm.get_torques()
        return self._select_joints(torques, joint_names)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pose = self.arm.get_pose()
        quat = tf.quaternion_from_euler(pose.rx, pose.ry, pose.rz)
        return np.array([pose.x, pose.y, pose.z], dtype=float), np.asarray(quat, dtype=float)

    def get_ee_pose_xyzrpy(self) -> np.ndarray:
        pose = self.arm.get_pose()
        return np.array([pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz], dtype=float)

    def wait_motion_done(self) -> bool:
        self.arm.wait_motion_done()
        return True

    def close(self, disable: bool = True) -> None:
        if self._closed:
            return
        try:
            if disable:
                self.disable()
                time.sleep(0.05)
        finally:
            self.arm.close()
            self._closed = True

    def _select_joints(
        self,
        values: List[float],
        joint_names: Optional[Union[str, List[str]]],
    ) -> Union[float, List[float]]:
        if joint_names is None:
            return values

        names = self.get_joint_names()
        if isinstance(joint_names, str):
            return values[names.index(joint_names)]
        return [values[names.index(name)] for name in joint_names]

    def __del__(self):
        try:
            self.close(disable=False)
        except Exception:
            pass
