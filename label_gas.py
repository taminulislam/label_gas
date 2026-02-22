"""
Gas Labeler - Interactive tool to draw and save gas region masks.

Workflow:
    1. Run the script
    2. Select the folder containing the images to label
    3. Draw boundary around gas region with left mouse
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
    Q / Escape  - Quit
"""

import cv2
import numpy as np
import os
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


class GasLabeler:
    """
    Interactive gas region labeling tool using OpenCV.

    Given a folder of images, this tool lets you draw regions on each image
    and saves:
      - A binary mask  → <parent>/masks/<filename>
      - A smooth color overlay → <parent>/overlays/<filename>

    Already-labeled images (mask already exists) are automatically skipped.
    """

    def __init__(self, frames_dir: Path):
        self.frames_dir = frames_dir
        parent_dir = frames_dir.parent

        # Output directories sit next to the selected folder
        self.masks_dir = parent_dir / "masks"
        self.overlays_dir = parent_dir / "overlays"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)

        # Collect images, skip already-labeled ones
        all_frames = sorted(
            [f for f in frames_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS],
            key=lambda x: numerical_sort(x.name)
        )
        labeled = {f.name for f in self.masks_dir.glob("*.png")}
        self.frames = [f for f in all_frames if f.name not in labeled]
        self.total = len(all_frames)

        # State
        self.idx = 0
        self.drawing = False
        self.erasing = False
        self.brush_size = 3
        self.last_point = None
        self.overlay_color = OVERLAY_COLOR
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

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fill_region(self):
        """Flood-fill all enclosed contours."""
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(self.mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
            self.mask = filled
            print(f"Filled {len(contours)} region(s)")
        else:
            print("No enclosed region found — draw a closed boundary first.")

    def _get_display(self):
        """Build the display frame with overlay + HUD text."""
        display = self.current_image.copy()

        # Color overlay
        overlay_layer = display.copy()
        overlay_layer[self.mask > 0] = self.overlay_color
        display = cv2.addWeighted(display, 0.55, overlay_layer, 0.45, 0)

        # Contour border
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, self.overlay_color, 2)

        # HUD: progress + brush size + filename
        labeled_count = len(list(self.masks_dir.glob("*.png")))
        hud = (
            f"{labeled_count + 1}/{self.total}  |  "
            f"Brush: {self.brush_size}  |  "
            f"{self.current_name}"
        )
        # Shadow for readability
        cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        return display

    def _create_smooth_overlay(self):
        """Smooth blended overlay (for saving — no hard border)."""
        display = self.current_image.copy()
        smooth_mask = cv2.GaussianBlur(self.mask, (15, 15), 0)
        alpha = smooth_mask.astype(float) / 255.0
        colored = np.zeros_like(display)
        colored[:] = self.overlay_color
        for c in range(3):
            display[:, :, c] = (
                (1 - alpha * 0.5) * display[:, :, c] +
                (alpha * 0.5) * colored[:, :, c]
            )
        return display.astype(np.uint8)

    def _load_frame(self):
        """Load the current frame into memory. Returns False when done."""
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
        """Save mask and overlay for the current frame."""
        if np.sum(self.mask) == 0:
            print("Nothing drawn — draw a region before saving.")
            return False

        mask_path = self.masks_dir / self.current_name
        overlay_path = self.overlays_dir / self.current_name

        cv2.imwrite(str(mask_path), self.mask)
        cv2.imwrite(str(overlay_path), self._create_smooth_overlay())
        print(f"Saved: {self.current_name}")
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self):
        if not self.frames:
            print("No unlabeled images found in the selected folder.")
            return

        cv2.namedWindow("Gas Labeler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gas Labeler", 1200, 800)
        cv2.setMouseCallback("Gas Labeler", self._mouse_callback)

        if not self._load_frame():
            print("Nothing to label.")
            return

        print(f"\nStarting — {len(self.frames)} image(s) to label.")

        while True:
            cv2.imshow("Gas Labeler", self._get_display())
            key = cv2.waitKey(30) & 0xFF

            if key == ord('f'):
                self._fill_region()

            elif key in (13, 32):   # Enter / Space → save & next
                if self._save():
                    self.idx += 1
                    if not self._load_frame():
                        print("\n=== All images labeled! ===")
                        break

            elif key == ord('n'):   # N → skip
                print(f"Skipped: {self.current_name}")
                self.idx += 1
                if not self._load_frame():
                    print("\n=== All images processed! ===")
                    break

            elif key == ord('c'):   # C → clear
                self.mask = np.zeros_like(self.mask)

            elif key in (ord('+'), ord('=')):
                self.brush_size = min(20, self.brush_size + 1)

            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 1)

            elif key in (ord('q'), 27):   # Q / Escape → quit
                print("Quit.")
                break

        cv2.destroyAllWindows()

        masks_done = len(list(self.masks_dir.glob("*.png")))
        overlays_done = len(list(self.overlays_dir.glob("*.png")))
        print(f"\nSession summary")
        print(f"  Masks    : {masks_done}")
        print(f"  Overlays : {overlays_done}")
        print(f"  Location : {self.masks_dir.parent}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def select_folder() -> Path | None:
    """Show a folder picker dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title="Select folder containing images to label")
    root.destroy()
    return Path(folder) if folder else None


def main():
    print("=== Gas Labeler ===\n")

    folder = select_folder()
    if folder is None:
        print("No folder selected. Exiting.")
        return

    # Validate that the folder contains images
    images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]
    if not images:
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            "No Images Found",
            f"No supported images found in:\n{folder}\n\n"
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
        root.destroy()
        print(f"No images found in: {folder}")
        return

    labeler = GasLabeler(folder)
    labeler.start()


if __name__ == "__main__":
    main()
