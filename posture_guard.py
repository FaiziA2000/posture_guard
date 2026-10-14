"""
PostureGuard — Real-time posture monitoring with screen-freeze overlay.
Runs silently in the background using your laptop camera.
"""

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import math
import sys
import os
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG — tweak these to your comfort
# ─────────────────────────────────────────────
CHECK_INTERVAL_SEC   = 0.1    # how often posture is evaluated (seconds)
BAD_POSTURE_GRACE    = 2.0    # seconds of bad posture before freeze triggers
GOOD_POSTURE_GRACE   = 1.5    # seconds of good posture before freeze lifts
SHOULDER_TILT_LIMIT  = 15     # max degrees shoulder tilt allowed
FORWARD_LEAN_RATIO   = 0.08   # nose-to-shoulder forward lean threshold (ratio of frame height)
CAMERA_INDEX         = 0      # 0 = default webcam

# ─────────────────────────────────────────────
#  POSTURE ANALYSER
# ─────────────────────────────────────────────
class PostureAnalyser:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def analyse(self, frame):
        """
        Returns (is_good, reason, landmarks_or_None)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, "No person detected", None  # None = uncertain

        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Key points (normalised 0-1)
        nose          = lm[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder= lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear      = lm[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear     = lm[self.mp_pose.PoseLandmark.RIGHT_EAR]

        # 1. Shoulder tilt — are shoulders level?
        dy = (left_shoulder.y - right_shoulder.y) * h
        dx = (left_shoulder.x - right_shoulder.x) * w
        tilt_deg = abs(math.degrees(math.atan2(dy, dx)))
        if tilt_deg > 90:
            tilt_deg = 180 - tilt_deg

        if tilt_deg > SHOULDER_TILT_LIMIT:
            return False, f"Shoulders tilted {tilt_deg:.0f}°", results.pose_landmarks

        # 2. Forward head lean — nose ahead of shoulders?
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        nose_vs_shoulder = mid_shoulder_y - nose.y   # positive = nose higher than shoulders (normal)

        if nose_vs_shoulder < FORWARD_LEAN_RATIO:
            return False, "Head leaning forward / slouching", results.pose_landmarks

        # 3. Head vertical — ears above shoulders?
        mid_ear_y = (left_ear.y + right_ear.y) / 2
        if mid_ear_y > mid_shoulder_y:
            return False, "Head dropped too low", results.pose_landmarks

        return True, "Good posture ✓", results.pose_landmarks

    def close(self):
        self.pose.close()


# ─────────────────────────────────────────────
#  SCREEN-FREEZE OVERLAY  (Tkinter full-screen)
# ─────────────────────────────────────────────
class FreezeOverlay:
    def __init__(self):
        self.root = None
        self.visible = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        time.sleep(0.4)          # let Tk start

    # ── called from any thread ──
    def show(self, reason="Correct your posture!"):
        with self._lock:
            self._pending_show = reason

    def hide(self):
        with self._lock:
            self._pending_hide = True

    # ── Tkinter loop (own thread) ──
    def _run_tk(self):
        self._pending_show = None
        self._pending_hide = False

        self.root = tk.Tk()
        self.root.withdraw()

        # Full-screen setup
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")

        # ── Canvas for blurred / dark overlay ──
        self.canvas = tk.Canvas(self.root, bg="#0a0a0f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Gradient-like dark overlay drawn on canvas
        self._draw_bg(sw, sh)

        # ── Card panel ──
        card_w, card_h = 520, 340
        cx, cy = sw // 2, sh // 2

        card = tk.Frame(self.root, bg="#12121e",
                        highlightbackground="#6c63ff",
                        highlightthickness=2,
                        bd=0)
        card.place(x=cx - card_w // 2, y=cy - card_h // 2,
                   width=card_w, height=card_h)

        # Icon (unicode spine emoji stand-in)
        icon_lbl = tk.Label(card, text="🧍", font=("Segoe UI Emoji", 52),
                            bg="#12121e", fg="#ffffff")
        icon_lbl.pack(pady=(28, 4))

        title_lbl = tk.Label(card, text="PostureGuard",
                             font=("Segoe UI", 13, "bold"),
                             bg="#12121e", fg="#6c63ff")
        title_lbl.pack()

        self.reason_lbl = tk.Label(card,
                                   text="Correct your posture!",
                                   font=("Segoe UI", 18, "bold"),
                                   bg="#12121e", fg="#ffffff",
                                   wraplength=460)
        self.reason_lbl.pack(pady=(14, 6))

        sub_lbl = tk.Label(card,
                           text="Sit upright — the screen will unlock automatically.",
                           font=("Segoe UI", 11),
                           bg="#12121e", fg="#8888aa",
                           wraplength=460)
        sub_lbl.pack()

        # Animated ring (canvas inside card)
        ring_size = 64
        self.ring_canvas = tk.Canvas(card, width=ring_size, height=ring_size,
                                     bg="#12121e", highlightthickness=0)
        self.ring_canvas.pack(pady=(18, 0))
        self._ring_angle = 0
        self._draw_ring(ring_size)

        self.root.after(50, self._poll)
        self.root.after(80, self._animate_ring)
        self.root.mainloop()

    def _draw_bg(self, w, h):
        # Create a dark semi-transparent feel with canvas rectangles
        steps = 8
        for i in range(steps):
            alpha_hex = format(int(20 + i * 5), '02x')
            self.canvas.create_rectangle(0, 0, w, h,
                                         fill="#0a0a0f", outline="")

    def _draw_ring(self, size):
        self.ring_canvas.delete("ring")
        pad = 6
        self.ring_canvas.create_arc(pad, pad, size - pad, size - pad,
                                    start=self._ring_angle,
                                    extent=270,
                                    outline="#6c63ff",
                                    width=4,
                                    style="arc",
                                    tags="ring")

    def _animate_ring(self):
        if self.visible:
            size = 64
            self._ring_angle = (self._ring_angle + 8) % 360
            self._draw_ring(size)
        self.root.after(40, self._animate_ring)

    def _poll(self):
        with self._lock:
            show_reason = self._pending_show
            do_hide = self._pending_hide
            self._pending_show = None
            self._pending_hide = False

        if show_reason and not self.visible:
            self.reason_lbl.config(text=show_reason)
            self.root.deiconify()
            self._fade_in()
            self.visible = True

        elif do_hide and self.visible:
            self._fade_out()
            self.visible = False

        self.root.after(100, self._poll)

    def _fade_in(self, alpha=0.0):
        if alpha <= 0.92:
            self.root.attributes("-alpha", alpha)
            self.root.after(20, lambda: self._fade_in(alpha + 0.07))

    def _fade_out(self, alpha=0.92):
        if alpha >= 0.0:
            self.root.attributes("-alpha", alpha)
            self.root.after(20, lambda: self._fade_out(alpha - 0.07))
        else:
            self.root.withdraw()
            self.root.attributes("-alpha", 0.0)


# ─────────────────────────────────────────────
#  TRAY ICON  (minimal — just a console fallback)
# ─────────────────────────────────────────────
def print_status(msg):
    # Overwrite same line in terminal
    sys.stdout.write(f"\r\033[K{msg}")
    sys.stdout.flush()


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════╗")
    print("║      PostureGuard  v1.0          ║")
    print("║  Camera monitoring active...     ║")
    print("║  Press Ctrl+C to quit            ║")
    print("╚══════════════════════════════════╝\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌  Cannot open camera. Check CAMERA_INDEX in config.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    analyser = PostureAnalyser()
    overlay  = FreezeOverlay()

    bad_since  = None   # timestamp when bad posture started
    good_since = None   # timestamp when good posture returned
    frozen     = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            is_good, reason, _ = analyser.analyse(frame)
            now = time.time()

            if is_good is None:
                # Camera can't see person — don't punish
                bad_since  = None
                good_since = None
                print_status("👁  Waiting to detect person...")
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            if not is_good:
                good_since = None
                if bad_since is None:
                    bad_since = now

                elapsed = now - bad_since
                remaining = max(0, BAD_POSTURE_GRACE - elapsed)

                if not frozen:
                    if elapsed >= BAD_POSTURE_GRACE:
                        overlay.show(reason)
                        frozen = True
                        print_status(f"🔴  FROZEN  |  {reason}")
                    else:
                        print_status(f"⚠️   Bad posture — freezing in {remaining:.1f}s  |  {reason}")
                else:
                    print_status(f"🔴  FROZEN  |  {reason}")

            else:
                bad_since = None
                if good_since is None:
                    good_since = now

                elapsed = now - good_since

                if frozen:
                    if elapsed >= GOOD_POSTURE_GRACE:
                        overlay.hide()
                        frozen = False
                        print_status("🟢  Good posture — screen unlocked!")
                    else:
                        remaining = GOOD_POSTURE_GRACE - elapsed
                        print_status(f"🟡  Hold still — unlocking in {remaining:.1f}s")
                else:
                    print_status("🟢  Posture looks great — keep it up!")

            time.sleep(CHECK_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n👋  PostureGuard stopped. Goodbye!")
    finally:
        cap.release()
        analyser.close()


if __name__ == "__main__":
    main()
