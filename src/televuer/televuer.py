from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import cv2
import os
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



class TeleVuer:
    def __init__(self, binocular: bool, use_hand_tracking: bool, img_shape, img_shm_name, cert_file=None, key_file=None, ngrok=False, webrtc=False, port=8012):
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

        self.webrtc = webrtc
        self.vuer.spawn(start=False)(self._main_handler)

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

        except:
            pass
    
    async def main_image_binocular(self, session, fps=60):
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
            display_image = self.img_array.copy()
            self._draw_hud_overlay(display_image)
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image[:, :self.img_width],
                        aspect=1.778,
                        height=1.5,
                        fixed=True,
                        position=[0, 1.5, -2],
                        # Layer bitmask: layers=1 → left eye only, layers=2 → right eye only.
                        layers=1,
                        format="jpeg",
                        quality=70,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        display_image[:, self.img_width:],
                        aspect=1.778,
                        height=1.5,
                        fixed=True,
                        position=[0, 1.5, -2],
                        layers=2,
                        format="jpeg",
                        quality=70,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # ‘jpeg’ encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
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
            display_image = self.img_array.copy()
            self._draw_hud_overlay(display_image)
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image,
                        aspect=1.778,
                        height=1.5,
                        fixed=True,
                        position=[0, 1.5, -2],
                        format="jpeg",
                        quality=85,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

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
