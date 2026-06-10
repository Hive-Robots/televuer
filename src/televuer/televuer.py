from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import ctypes
import cv2
import os
import signal
import time as _time
from pathlib import Path


def _draw_text_panel(frame: np.ndarray, lines: list, x: int, y: int,
                     color=(255, 255, 255)) -> None:
    """Draw a multi-line monospace text block with a semi-transparent background."""
    font, scale, thick, line_h = cv2.FONT_HERSHEY_PLAIN, 1.2, 1, 26
    pad = 5
    max_w = max((cv2.getTextSize(ln, font, scale, thick)[0][0] for ln in lines), default=0)
    x0, y0 = x - pad, y - 14
    x1, y1 = x + max_w + pad, y + len(lines) * line_h + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (x, y + i * line_h), font, scale, color, thick, cv2.LINE_AA)


def _obj_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _panel_position(panel, fallback: list[float]) -> list[float]:
    x = _obj_attr(panel, "x")
    y = _obj_attr(panel, "y")
    z = _obj_attr(panel, "z")
    if x is None or y is None or z is None:
        return fallback
    return [float(x), float(y), float(z)]


def _fallback_panel_position(position: str) -> list[float]:
    positions = {
        "left": [-1.35, 1.35, -2.05],
        "right": [1.35, 1.35, -2.05],
        "top": [0.0, 2.15, -2.05],
        "tl": [-0.85, 2.05, -1.9],
        "tr": [0.85, 2.05, -1.9],
        "bl": [-0.85, 0.65, -1.9],
        "br": [0.85, 0.65, -1.9],
    }
    return positions.get(position, [0.0, 1.35, -2.0])


class TeleVuer:
    def __init__(self, binocular: bool, use_hand_tracking: bool, img_shape, img_shm_name, cert_file=None, key_file=None, ngrok=False, webrtc=False, port=8012,
                 cam_shm_buffers=None, cam_segments=None, cam_layout=None):
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages the shared memory for image and pose data.

        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
        :param img_shape: tuple, shape of the image (height, width, channels).
        :param img_shm_name: str, name of the shared memory for the image.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.
        :param ngrok: bool, whether to use ngrok for tunneling.
        :param port: int, port number for the Vuer server (default: 8012).
        """
        self.binocular = binocular
        self.use_hand_tracking = use_hand_tracking
        self.img_height = img_shape[0]
        if self.binocular:
            self.img_width  = img_shape[1] // 2
        else:
            self.img_width  = img_shape[1]
        
        current_module_dir = Path(__file__).resolve().parent.parent.parent
        if cert_file is None:
            cert_file = os.path.join(current_module_dir, "cert.pem")
        if key_file is None:
            key_file = os.path.join(current_module_dir, "key.pem")

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', port=port, queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(host='0.0.0.0', port=port, cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)
        self.cam_shm_buffers = cam_shm_buffers or {}
        self.cam_segments = {_obj_attr(segment, "name"): segment for segment in (cam_segments or [])}
        self.cam_layout = cam_layout

        self.webrtc = webrtc
        self.vuer.spawn(start=False)(self._main_handler)

        # Geometry cache: only send position/rotation to the browser when they change.
        # React's useLayoutEffect compares array props by reference (Object.is), so a new
        # array with the same values still triggers the effect and can fire mesh.position.set()
        # mid-frame between left/right eye renders, causing the "opposite-direction" jitter.
        # By omitting position/rotation when unchanged, the Three.js mesh retains its world
        # position without the effect re-firing.
        self._screen_geom_cache: dict = {}   # key → (pos_tuple, rot_tuple)
        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_pinch_state_shared = Value('b', False, lock=True)
            self.left_pinch_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)

            self.right_pinch_state_shared = Value('b', False, lock=True)
            self.right_pinch_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
        else:
            self.left_trigger_state_shared = Value('b', False, lock=True)
            self.left_trigger_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.left_thumbstick_state_shared = Value('b', False, lock=True)
            self.left_thumbstick_value_shared = Array('d', 2, lock=True)
            self.left_aButton_shared = Value('b', False, lock=True)
            self.left_bButton_shared = Value('b', False, lock=True)

            self.right_trigger_state_shared = Value('b', False, lock=True)
            self.right_trigger_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.right_thumbstick_state_shared = Value('b', False, lock=True)
            self.right_thumbstick_value_shared = Array('d', 2, lock=True)
            self.right_aButton_shared = Value('b', False, lock=True)
            self.right_bButton_shared = Value('b', False, lock=True)

        # HUD — written by main loop via TeleVuerWrapper, read by spawned async tasks.
        self.hud_reveal_shared       = Value('b', False, lock=True)
        self.hud_ctrl_map_shared_vis = Value('b', False, lock=True)
        self.hud_recording_shared    = Value('b', False, lock=True)   # also drives rec dot
        self.hud_task_name_shared    = Array('c', 256, lock=True)
        self.hud_arms_shared         = Array('c', 32,  lock=True)
        self.hud_ep_good_shared      = Value('i', 0, lock=True)
        self.hud_ep_bad_shared       = Value('i', 0, lock=True)
        self.hud_ep_review_shared    = Value('i', 0, lock=True)
        self.hud_left_preset_shared  = Array('c', 64, lock=True)
        self.hud_right_preset_shared = Array('c', 64, lock=True)
        self.hud_ctrl_map_shared     = Array('c', 4096, lock=True)
        self.hud_notify_text_shared  = Array('c', 128, lock=True)
        self.hud_notify_ts_shared    = Value('d', 0.0, lock=True)

        self.last_pose_t = Value('d', 0.0, lock=False)

        # Kill any stale process holding this port (e.g., from a previous run that segfaulted
        # and bypassed daemon-process cleanup). This ensures the new vuer server can bind.
        import subprocess as _sp
        try:
            _sp.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=2.0)
            _time.sleep(0.3)
        except Exception:
            pass

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()

    async def _main_handler(self, session):
        """Single socket handler that runs the image stream and HUD tasks concurrently.

        vuer.spawn() stores only ONE socket_handler — calling it multiple times
        overwrites the previous. We use session.spawn_task() to run the HUD
        coroutines as independent asyncio tasks, then await the image coroutine.
        HUD tasks are cancelled in a finally block so they don't outlive the session.
        """
        if self.binocular and not self.webrtc:
            await self.main_image_binocular(session)
        elif not self.binocular and not self.webrtc:
            await self.main_image_monocular(session)
        else:
            await self.main_image_webrtc(session)

    def vuer_run(self):
        # Ask the kernel to send SIGTERM to this process when the parent dies,
        # even if the parent is killed by a signal (e.g. segfault) that bypasses
        # Python's atexit/daemon-process cleanup.
        try:
            ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG=1
        except Exception:
            pass
        try:
            self.vuer.run()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Vuer encountered an error: {e}")

    async def on_cam_move(self, event, session, fps=60):
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        try:
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]

            left_controller_state = event.value["leftState"]
            right_controller_state = event.value["rightState"]

            def extract_controller_states(state_dict, prefix):
                # trigger
                with getattr(self, f"{prefix}_trigger_state_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_state_shared").value = bool(state_dict.get("trigger", False))
                with getattr(self, f"{prefix}_trigger_value_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_value_shared").value = float(state_dict.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"{prefix}_thumbstick_state_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_state_shared").value = bool(state_dict.get("thumbstick", False))
                with getattr(self, f"{prefix}_thumbstick_value_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_value_shared")[:] = state_dict.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"{prefix}_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_aButton_shared").value = bool(state_dict.get("aButton", False))
                with getattr(self, f"{prefix}_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_bButton_shared").value = bool(state_dict.get("bButton", False))

            extract_controller_states(left_controller_state, "left")
            extract_controller_states(right_controller_state, "right")
            self.last_pose_t.value = _time.perf_counter()
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand_state = event.value["leftState"]
            right_hand_state = event.value["rightState"]

            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]

            def extract_hand_states(state_dict, prefix):
                # pinch
                with getattr(self, f"{prefix}_pinch_state_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_state_shared").value = bool(state_dict.get("pinch", False))
                with getattr(self, f"{prefix}_pinch_value_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_value_shared").value = float(state_dict.get("pinchValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            extract_hand_states(left_hand_state, "left")
            extract_hand_states(right_hand_state, "right")
            self.last_pose_t.value = _time.perf_counter()
        except:
            pass
    
    async def main_image_binocular(self, session, fps=60):
        self._screen_geom_cache.clear()
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            try:
                session.upsert(self._camera_screen_elements(binocular=True), to="bgChildren")
            except AssertionError:
                break
            # ‘jpeg’ encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        self._screen_geom_cache.clear()
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            try:
                session.upsert(self._camera_screen_elements(binocular=False), to="bgChildren")
            except AssertionError:
                break
            await asyncio.sleep(0.033)

    async def main_image_webrtc(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    showLeft=False,
                    showRight=False
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    showLeft=False,
                    showRight=False,
                )
            )
    
        session.upsert(
            WebRTCVideoPlane(
            # WebRTCStereoVideoPlane(
                src="https://10.0.7.49:8080/offer",
                iceServer={},
                key="webrtc",
                aspect=1.778,
                height = 7,
            ),
            to="bgChildren",
        )
        while True:
            await asyncio.sleep(1)

    def _active_layout(self):
        return self.cam_layout

    def _camera_screen_elements(self, binocular: bool):
        if self.cam_layout is None or not self.cam_shm_buffers:
            frame = self.img_array.copy()
            self._draw_hud_overlay(frame)
            return self._image_elements(
                frame,
                key="background",
                height=1.15,
                position=[0.0, 1.0, -2.0],
                binocular=binocular,
                quality=85,
            )

        layout = self._active_layout()
        main_name = _obj_attr(layout, "main")

        elements = []
        main_segment = self.cam_segments.get(main_name)
        main_buffer = self._segment_buffer(main_segment)
        if main_buffer is not None:
            main = main_buffer.copy()
            self._draw_hud_overlay(main)
            elements.extend(
                self._image_elements(
                    main,
                    key=f"camera-{main_name}",
                    height=float(_obj_attr(layout, "main_height", 1.15)),
                    position=list(_obj_attr(layout, "main_position", [0.0, 1.0, -2.0])),
                    binocular=binocular and bool(_obj_attr(main_segment, "binocular", False)),
                    quality=int(_obj_attr(layout, "main_quality", 85)),
                )
            )

        default_panel_quality = int(_obj_attr(layout, "panel_quality", 70))
        for panel in _obj_attr(layout, "panels", []):
            segment_name = _obj_attr(panel, "segment")
            segment = self.cam_segments.get(segment_name)
            buffer = self._segment_buffer(segment)
            if buffer is None:
                continue
            position_name = _obj_attr(panel, "position")
            height = _obj_attr(panel, "height")
            if height is None:
                height = max(0.35, float(_obj_attr(panel, "scale", 0.6)))
            raw_rot = _obj_attr(panel, "rotation")
            rotation = list(raw_rot) if raw_rot is not None else None
            panel_quality = _obj_attr(panel, "quality")
            if panel_quality is None:
                panel_quality = default_panel_quality
            elements.extend(
                self._image_elements(
                    buffer.copy(),
                    key=f"camera-{segment_name}",
                    height=float(height),
                    position=_panel_position(panel, _fallback_panel_position(position_name)),
                    binocular=binocular and bool(_obj_attr(segment, "binocular", False)),
                    quality=int(panel_quality),
                    rotation=rotation,
                )
            )

        return elements

    def _segment_buffer(self, segment):
        if segment is None:
            return None
        return self.cam_shm_buffers.get(_obj_attr(segment, "shm_name"))

    def _image_elements(self, frame, *, key: str, height: float, position: list[float], binocular: bool, quality: int, rotation: list[float] | None = None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        geom_sig = (tuple(position), tuple(rotation) if rotation is not None else None)

        def _geom_kwargs(k: str) -> dict:
            """Return position/rotation kwargs only when they changed since last send.
            Omitting them on unchanged frames prevents React's useLayoutEffect from
            calling mesh.position.set() every frame and causing mid-render jitter."""
            if self._screen_geom_cache.get(k) != geom_sig:
                self._screen_geom_cache[k] = geom_sig
                kw = {"position": position}
                if rotation is not None:
                    kw["rotation"] = rotation
                return kw
            return {}

        if binocular:
            split = rgb.shape[1] // 2
            aspect = split / rgb.shape[0]
            return [
                ImageBackground(
                    rgb[:, :split],
                    aspect=aspect,
                    height=height,
                    fixed=True,
                    layers=1,
                    format="jpeg",
                    quality=quality,
                    key=f"{key}-left",
                    interpolate=True,
                    **_geom_kwargs(f"{key}-left"),
                ),
                ImageBackground(
                    rgb[:, split:],
                    aspect=aspect,
                    height=height,
                    fixed=True,
                    layers=2,
                    format="jpeg",
                    quality=quality,
                    key=f"{key}-right",
                    interpolate=True,
                    **_geom_kwargs(f"{key}-right"),
                ),
            ]

        aspect = rgb.shape[1] / rgb.shape[0]
        return [
            ImageBackground(
                rgb,
                aspect=aspect,
                height=height,
                fixed=True,
                format="jpeg",
                quality=quality,
                key=key,
                interpolate=True,
                **_geom_kwargs(key),
            )
        ]

    # ==================== HUD ====================
    def _draw_hud_overlay(self, frame: np.ndarray) -> None:
        """Draw all HUD elements onto frame in-place (BGR)."""
        w = frame.shape[1]

        # Recording dot — always on
        recording = bool(self.hud_recording_shared.value)
        cx, cy, r = 30, 30, 14
        cv2.circle(frame, (cx, cy), r, (0, 0, 220) if recording else (120, 120, 120), -1)
        if recording:
            cv2.putText(frame, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 220), 2, cv2.LINE_AA)

        # Transient notification — yellow, centred near top
        msg = bytes(self.hud_notify_text_shared[:]).decode("utf-8", "replace").rstrip("\x00")
        if msg and (_time.time() - self.hud_notify_ts_shared.value) < 2.0:
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(frame, msg, (w // 2 - tw // 2, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (57, 255, 20), 2, cv2.LINE_AA)

        # Status panel — hold-to-reveal, left side
        if bool(self.hud_reveal_shared.value):
            task  = bytes(self.hud_task_name_shared[:]).decode("utf-8", "replace").rstrip("\x00")
            arms  = bytes(self.hud_arms_shared[:]).decode("utf-8", "replace").rstrip("\x00")
            g, b, rv = (self.hud_ep_good_shared.value, self.hud_ep_bad_shared.value,
                        self.hud_ep_review_shared.value)
            lp    = bytes(self.hud_left_preset_shared[:]).decode("utf-8", "replace").rstrip("\x00")
            rp    = bytes(self.hud_right_preset_shared[:]).decode("utf-8", "replace").rstrip("\x00")
            lines = [
                "REC" if recording else "READY",
                f"Task: {task}",
                f"Arms: {arms}",
                f"Ep: {g+b+rv}  ({g}g / {b}b / {rv}r)",
                f"L preset: {lp}",
                f"R preset: {rp}",
            ]
            _draw_text_panel(frame, lines, x=10, y=80)

        # Ctrl-map panel — hold-to-reveal, right side
        if bool(self.hud_ctrl_map_shared_vis.value):
            ctrl = bytes(self.hud_ctrl_map_shared[:]).decode("utf-8", "replace").rstrip("\x00")
            _draw_text_panel(frame, ctrl.split("\n"), x=w - 310, y=80, color=(180, 255, 180))

    # ==================== common data ====================

    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch_state(self):
        """bool, whether left hand is pinching."""
        with self.left_pinch_state_shared.get_lock():
            return self.left_pinch_state_shared.value

    @property
    def left_hand_pinch_value(self):
        """float, pinch strength of left hand."""
        with self.left_pinch_value_shared.get_lock():
            return self.left_pinch_value_shared.value

    @property
    def left_hand_squeeze_state(self):
        """bool, whether left hand is squeezing."""
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_hand_squeeze_value(self):
        """float, squeeze strength of left hand."""
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def right_hand_pinch_state(self):
        """bool, whether right hand is pinching."""
        with self.right_pinch_state_shared.get_lock():
            return self.right_pinch_state_shared.value

    @property
    def right_hand_pinch_value(self):
        """float, pinch strength of right hand."""
        with self.right_pinch_value_shared.get_lock():
            return self.right_pinch_value_shared.value

    @property
    def right_hand_squeeze_state(self):
        """bool, whether right hand is squeezing."""
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_hand_squeeze_value(self):
        """float, squeeze strength of right hand."""
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    # ==================== Controller Data ====================
    @property
    def left_controller_trigger_state(self):
        """bool, left controller trigger pressed or not."""
        with self.left_trigger_state_shared.get_lock():
            return self.left_trigger_state_shared.value

    @property
    def left_controller_trigger_value(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self.left_trigger_value_shared.get_lock():
            return self.left_trigger_value_shared.value

    @property
    def left_controller_squeeze_state(self):
        """bool, left controller squeeze pressed or not."""
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_controller_squeeze_value(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def left_controller_thumbstick_state(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self.left_thumbstick_state_shared.get_lock():
            return self.left_thumbstick_state_shared.value

    @property
    def left_controller_thumbstick_value(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self.left_thumbstick_value_shared.get_lock():
            return np.array(self.left_thumbstick_value_shared[:])

    @property
    def left_controller_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self.left_aButton_shared.get_lock():
            return self.left_aButton_shared.value

    @property
    def left_controller_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self.left_bButton_shared.get_lock():
            return self.left_bButton_shared.value

    @property
    def right_controller_trigger_state(self):
        """bool, right controller trigger pressed or not."""
        with self.right_trigger_state_shared.get_lock():
            return self.right_trigger_state_shared.value

    @property
    def right_controller_trigger_value(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self.right_trigger_value_shared.get_lock():
            return self.right_trigger_value_shared.value

    @property
    def right_controller_squeeze_state(self):
        """bool, right controller squeeze pressed or not."""
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_controller_squeeze_value(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    @property
    def right_controller_thumbstick_state(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self.right_thumbstick_state_shared.get_lock():
            return self.right_thumbstick_state_shared.value

    @property
    def right_controller_thumbstick_value(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self.right_thumbstick_value_shared.get_lock():
            return np.array(self.right_thumbstick_value_shared[:])

    @property
    def right_controller_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self.right_aButton_shared.get_lock():
            return self.right_aButton_shared.value

    @property
    def right_controller_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self.right_bButton_shared.get_lock():
            return self.right_bButton_shared.value

    def consume_thumbstick_data(self):
        """Read and clear all thumbstick state/value atomically per variable.

        Returns (left_state, left_value, right_state, right_value).
        Clearing on read ensures stale data produces zero commands when the VR
        connection stops sending updates.
        """
        with self.left_thumbstick_state_shared.get_lock():
            left_state = bool(self.left_thumbstick_state_shared.value)
            self.left_thumbstick_state_shared.value = False
        with self.left_thumbstick_value_shared.get_lock():
            left_value = np.array(self.left_thumbstick_value_shared[:])
            self.left_thumbstick_value_shared[:] = [0.0, 0.0]
        with self.right_thumbstick_state_shared.get_lock():
            right_state = bool(self.right_thumbstick_state_shared.value)
            self.right_thumbstick_state_shared.value = False
        with self.right_thumbstick_value_shared.get_lock():
            right_value = np.array(self.right_thumbstick_value_shared[:])
            self.right_thumbstick_value_shared[:] = [0.0, 0.0]
        return left_state, left_value, right_state, right_value
