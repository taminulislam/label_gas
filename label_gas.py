"""
Gas Labeler - Interactive tool to draw and save gas region masks.

Workflow:
    1. Run the script — a home window appears
    2. Click "Select Folder" and choose your images folder
    3. Draw a boundary around the gas region with the left mouse button
    4. Press F to fill the enclosed region
    5. Press Enter/Space to save mask + overlay and move to next image
    6. Outputs saved to masks/ and overlays/ next to the selected folder

Controls:
    Left Mouse  - Draw boundary
    Right Mouse - Erase
    Scroll / +/- - Adjust brush size
    F           - Fill enclosed region
    Enter/Space - Save & next
    N           - Skip (no label)
    C           - Clear mask
    Q / Escape  - Quit labeling
"""

import cv2
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path


SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
OVERLAY_COLOR = (250, 206, 135)  # Light amber in BGR


def numerical_sort(value):
    """Sort strings containing numbers in natural order."""
    parts = re.split(r'(\d+)', str(value))
    return [int(part) if part.isdigit() else part for part in parts]


# ──────────────────────────────────────────────────────────────────────────────
# OpenCV labeling engine
# ──────────────────────────────────────────────────────────────────────────────

class GasLabeler:
    """
    OpenCV-based gas region labeling tool.

    Saves for each labeled image:
      - Binary mask    → <parent>/masks/<filename>
      - Smooth overlay → <parent>/overlays/<filename>

    Already-labeled images are automatically skipped on re-launch.
    """

    def __init__(self, frames_dir: Path):
        self.frames_dir = frames_dir
        parent_dir = frames_dir.parent

        self.masks_dir = parent_dir / "masks"
        self.overlays_dir = parent_dir / "overlays"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)

        all_frames = sorted(
            [f for f in frames_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS],
            key=lambda x: numerical_sort(x.name)
        )
        labeled = {f.name for f in self.masks_dir.glob("*.png")}
        self.frames = [f for f in all_frames if f.name not in labeled]
        self.total = len(all_frames)

        self.idx = 0
        self.drawing = False
        self.erasing = False
        self.brush_size = 3
        self.last_point = None
        self.mask = None
        self.current_image = None
        self.current_name = None

        print(f"Selected folder : {self.frames_dir}")
        print(f"Masks output    : {self.masks_dir}")
        print(f"Overlays output : {self.overlays_dir}")
        print(f"Total images    : {self.total}")
        print(f"Already labeled : {len(labeled)}")
        print(f"To label        : {len(self.frames)}")
        print()
        print("=== CONTROLS ===")
        print("LEFT MOUSE  : Draw    | RIGHT MOUSE : Erase")
        print("F           : Fill    | C           : Clear")
        print("ENTER/SPACE : Save    | N           : Skip")
        print("+/-/Scroll  : Brush   | Q/ESC       : Quit")

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.erasing = False
            self.last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
            self.drawing = False
            self.last_point = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False
            self.last_point = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.last_point is not None:
                cv2.line(self.mask, self.last_point, (x, y), 255, self.brush_size * 2)
                self.last_point = (x, y)
            elif self.erasing and self.last_point is not None:
                cv2.line(self.mask, self.last_point, (x, y), 0, self.brush_size * 4)
                self.last_point = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self.brush_size = max(1, min(20, self.brush_size + delta))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fill_region(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(self.mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
            self.mask = filled
            print(f"Filled {len(contours)} region(s)")
        else:
            print("No enclosed region found — draw a closed boundary first.")

    def _get_display(self):
        display = self.current_image.copy()
        overlay_layer = display.copy()
        overlay_layer[self.mask > 0] = OVERLAY_COLOR
        display = cv2.addWeighted(display, 0.55, overlay_layer, 0.45, 0)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, OVERLAY_COLOR, 2)

        labeled_count = len(list(self.masks_dir.glob("*.png")))
        hud = (
            f"{labeled_count + 1}/{self.total}  |  "
            f"Brush: {self.brush_size}  |  "
            f"{self.current_name}"
        )
        cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    def _create_smooth_overlay(self):
        display = self.current_image.copy()
        smooth_mask = cv2.GaussianBlur(self.mask, (15, 15), 0)
        alpha = smooth_mask.astype(float) / 255.0
        colored = np.zeros_like(display)
        colored[:] = OVERLAY_COLOR
        for c in range(3):
            display[:, :, c] = (
                (1 - alpha * 0.5) * display[:, :, c] +
                (alpha * 0.5) * colored[:, :, c]
            )
        return display.astype(np.uint8)

    def _load_frame(self):
        while self.idx < len(self.frames):
            frame_path = self.frames[self.idx]
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"Skipping unreadable file: {frame_path.name}")
                self.idx += 1
                continue
            self.current_name = frame_path.name
            self.current_image = img
            h, w = img.shape[:2]
            self.mask = np.zeros((h, w), dtype=np.uint8)
            return True
        return False

    def _save(self):
        if np.sum(self.mask) == 0:
            print("Nothing drawn — draw a region before saving.")
            return False
        cv2.imwrite(str(self.masks_dir / self.current_name), self.mask)
        cv2.imwrite(str(self.overlays_dir / self.current_name), self._create_smooth_overlay())
        print(f"Saved: {self.current_name}")
        return True

    # ── Main loop ─────────────────────────────────────────────────────────────

    def start(self):
        if not self.frames:
            print("No unlabeled images found.")
            return

        cv2.namedWindow("Gas Labeler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gas Labeler", 1200, 800)
        cv2.setMouseCallback("Gas Labeler", self._mouse_callback)

        if not self._load_frame():
            return

        print(f"\nStarting — {len(self.frames)} image(s) to label.")

        while True:
            cv2.imshow("Gas Labeler", self._get_display())
            key = cv2.waitKey(30) & 0xFF

            if key == ord('f'):
                self._fill_region()
            elif key in (13, 32):           # Enter / Space
                if self._save():
                    self.idx += 1
                    if not self._load_frame():
                        print("\n=== All images labeled! ===")
                        break
            elif key == ord('n'):           # Skip
                print(f"Skipped: {self.current_name}")
                self.idx += 1
                if not self._load_frame():
                    print("\n=== All images processed! ===")
                    break
            elif key == ord('c'):           # Clear
                self.mask = np.zeros_like(self.mask)
            elif key in (ord('+'), ord('=')):
                self.brush_size = min(20, self.brush_size + 1)
            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 1)
            elif key in (ord('q'), 27):     # Quit
                print("Quit.")
                break

        cv2.destroyAllWindows()
        return len(list(self.masks_dir.glob("*.png"))), len(list(self.overlays_dir.glob("*.png")))


# ──────────────────────────────────────────────────────────────────────────────
# Tkinter home window
# ──────────────────────────────────────────────────────────────────────────────

class GasLabelerApp:
    """Dark-themed home window — select a folder to begin labeling."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gas Labeler")
        self.root.configure(bg='#1a1a1a')
        self.root.state('zoomed')

        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(
            main_frame,
            text="Gas Labeler",
            font=('Arial', 32, 'bold'),
            fg='#ffffff',
            bg='#1a1a1a',
        ).pack(pady=(150, 20))

        # ── Subtitle / controls hint ───────────────────────────────────────────
        tk.Label(
            main_frame,
            text="Draw = Left Mouse  |  Fill = F  |  Save = Enter  |  Skip = N  |  Erase = Right Mouse",
            font=('Arial', 14),
            fg='#888888',
            bg='#1a1a1a',
        ).pack(pady=(0, 50))

        # ── Select Folder button ───────────────────────────────────────────────
        tk.Button(
            main_frame,
            text="Select Folder",
            font=('Arial', 16),
            fg='#ffffff',
            bg='#4a4a4a',
            activebackground='#5a5a5a',
            activeforeground='#ffffff',
            relief=tk.FLAT,
            padx=40,
            pady=15,
            cursor='hand2',
            command=self._on_select_folder,
        ).pack(pady=20)

    # ── Button handler ────────────────────────────────────────────────────────

    def _on_select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing images to label")
        if not folder:
            return  # cancelled — stay on home

        folder_path = Path(folder)
        images = [f for f in folder_path.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]

        if not images:
            messagebox.showwarning(
                "No Images Found",
                f"No supported images found in:\n{folder}\n\n"
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
            return  # stay on home

        # Hide home window, run OpenCV labeling (blocking)
        self.root.withdraw()

        labeler = GasLabeler(folder_path)
        result = labeler.start()

        # Show summary then return to home
        if result is not None:
            masks_done, overlays_done = result
            messagebox.showinfo(
                "Session Complete",
                f"Labeling session finished!\n\n"
                f"Masks saved    : {masks_done}\n"
                f"Overlays saved : {overlays_done}\n\n"
                f"Location: {labeler.masks_dir.parent}"
            )

        self.root.deiconify()
        self.root.state('zoomed')

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = GasLabelerApp()
    app.run()
