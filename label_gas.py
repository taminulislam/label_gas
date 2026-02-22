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
    Left Mouse   - Draw boundary
    Right Mouse  - Erase
    Scroll / +/- - Adjust brush size
    F            - Fill enclosed region
    Enter/Space  - Save & next
    N            - Skip (no label)
    C            - Clear mask
    Escape       - Back to home
"""

import cv2
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk


SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Sky-blue overlay colour
OVERLAY_COLOR_RGB = (135, 206, 250)   # RGB  – used for PIL display
OVERLAY_COLOR_BGR = (250, 206, 135)   # BGR  – used for OpenCV save


def numerical_sort(value):
    """Sort strings that contain numbers in natural (human) order."""
    parts = re.split(r'(\d+)', str(value))
    return [int(p) if p.isdigit() else p for p in parts]


# ──────────────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────────────

class GasLabelerApp:
    """
    Two-page tkinter app.

    Page 1 – Home:     title, controls hint, "Select Folder" button.
    Page 2 – Labeling: tkinter Canvas for drawing + keyboard shortcuts.

    OpenCV is used only for mask arithmetic and file I/O (no cv2.imshow).
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gas Labeler")
        self.root.configure(bg='#1a1a1a')
        self.root.state('zoomed')

        # ── Labeling state ─────────────────────────────────────────────────
        self._frames:      list       = []
        self._idx:         int        = 0
        self._total:       int        = 0
        self._mask:        np.ndarray = None   # type: ignore
        self._orig_image:  np.ndarray = None   # type: ignore  (RGB)
        self._current_name: str       = None   # type: ignore
        self._masks_dir:   Path       = None   # type: ignore
        self._overlays_dir: Path      = None   # type: ignore

        self._drawing      = False
        self._erasing      = False
        self._last_pt      = None   # last point in image coords
        self.brush_size    = 3

        self._photo        = None   # PhotoImage reference (prevent GC)
        self._scale        = None   # canvas→image scale factor
        self._offset_x     = 0
        self._offset_y     = 0

        # ── Build UI ───────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg='#1a1a1a')
        main.pack(fill=tk.BOTH, expand=True)

        self._build_home_page(main)
        self._build_labeling_page(main)
        self._show_home()

    # ── Home page ──────────────────────────────────────────────────────────

    def _build_home_page(self, parent):
        self._home_frame = tk.Frame(parent, bg='#1a1a1a')

        tk.Label(
            self._home_frame,
            text="Gas Labeler",
            font=('Arial', 32, 'bold'),
            fg='#ffffff',
            bg='#1a1a1a',
        ).pack(pady=(150, 20))

        tk.Label(
            self._home_frame,
            text="Draw = Left Mouse  |  Fill = F  |  Save = Enter  |  Skip = N  |  Erase = Right Mouse",
            font=('Arial', 14),
            fg='#888888',
            bg='#1a1a1a',
        ).pack(pady=(0, 50))

        tk.Button(
            self._home_frame,
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

    # ── Labeling page ──────────────────────────────────────────────────────

    def _build_labeling_page(self, parent):
        self._labeling_frame = tk.Frame(parent, bg='#1a1a1a')

        # Progress counter
        self._progress_var = tk.StringVar()
        tk.Label(
            self._labeling_frame,
            textvariable=self._progress_var,
            font=('Arial', 18, 'bold'),
            fg='#00ff88',
            bg='#1a1a1a',
        ).pack(pady=10)

        # Controls hint
        tk.Label(
            self._labeling_frame,
            text=(
                "Left Mouse = Draw  |  Right Mouse = Erase  |  F = Fill  |  "
                "C = Clear  |  Enter/Space = Save  |  N = Skip  |  Escape = Home"
            ),
            font=('Arial', 11),
            fg='#888888',
            bg='#1a1a1a',
        ).pack(pady=5)

        # Drawing canvas
        self._canvas = tk.Canvas(
            self._labeling_frame,
            bg='#1a1a1a',
            highlightthickness=0,
            cursor='crosshair',
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self._status_var = tk.StringVar()
        tk.Label(
            self._labeling_frame,
            textvariable=self._status_var,
            font=('Arial', 11),
            fg='#cccccc',
            bg='#2a2a2a',
            anchor='center',
            pady=5,
        ).pack(fill=tk.X)

        # ── Mouse bindings ────────────────────────────────────────────────
        self._canvas.bind('<ButtonPress-1>',   self._on_draw_start)
        self._canvas.bind('<B1-Motion>',        self._on_draw_move)
        self._canvas.bind('<ButtonRelease-1>', self._on_draw_stop)
        self._canvas.bind('<ButtonPress-3>',   self._on_erase_start)
        self._canvas.bind('<B3-Motion>',        self._on_erase_move)
        self._canvas.bind('<ButtonRelease-3>', self._on_erase_stop)
        self._canvas.bind('<MouseWheel>',       self._on_scroll)
        self._canvas.bind('<Configure>',        self._on_canvas_resize)

        # ── Keyboard bindings ─────────────────────────────────────────────
        for key in ('<f>', '<F>'):
            self.root.bind(key, lambda e: self._fill_region())
        for key in ('<Return>', '<space>'):
            self.root.bind(key, lambda e: self._save_and_next())
        for key in ('<n>', '<N>'):
            self.root.bind(key, lambda e: self._skip())
        for key in ('<c>', '<C>'):
            self.root.bind(key, lambda e: self._clear_mask())
        for key in ('<plus>', '<equal>'):
            self.root.bind(key, lambda e: self._adjust_brush(1))
        self.root.bind('<minus>', lambda e: self._adjust_brush(-1))
        self.root.bind('<Escape>', lambda e: self._show_home())

    # ── Navigation ────────────────────────────────────────────────────────

    def _show_home(self):
        self._labeling_frame.pack_forget()
        self._home_frame.pack(fill=tk.BOTH, expand=True)
        self.root.title("Gas Labeler")

    def _show_labeling(self):
        self._home_frame.pack_forget()
        self._labeling_frame.pack(fill=tk.BOTH, expand=True)

    # ── Folder selection ──────────────────────────────────────────────────

    def _on_select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing images to label")
        if not folder:
            return

        folder_path = Path(folder)
        images = [f for f in folder_path.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]

        if not images:
            messagebox.showwarning(
                "No Images Found",
                f"No supported images found in:\n{folder}\n\n"
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
            return

        self._load_folder(folder_path)

        if not self._frames:
            messagebox.showinfo(
                "All Labeled",
                "All images in this folder are already labeled!"
            )
            return

        self._show_labeling()
        self._load_current_frame()
        # Defer display until canvas has been laid out
        self.root.after(50, self._update_display)

    def _load_folder(self, folder_path: Path):
        parent_dir = folder_path.parent
        self._masks_dir    = parent_dir / "masks"
        self._overlays_dir = parent_dir / "overlays"
        self._masks_dir.mkdir(parents=True, exist_ok=True)
        self._overlays_dir.mkdir(parents=True, exist_ok=True)

        all_frames = sorted(
            [f for f in folder_path.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS],
            key=lambda x: numerical_sort(x.name)
        )
        labeled = {f.name for f in self._masks_dir.glob("*.png")}
        self._frames = [f for f in all_frames if f.name not in labeled]
        self._total  = len(all_frames)
        self._idx    = 0

        print(f"Selected folder : {folder_path}")
        print(f"Masks output    : {self._masks_dir}")
        print(f"Overlays output : {self._overlays_dir}")
        print(f"Total images    : {self._total}")
        print(f"Already labeled : {len(labeled)}")
        print(f"To label        : {len(self._frames)}")

    # ── Frame loading ─────────────────────────────────────────────────────

    def _load_current_frame(self) -> bool:
        while self._idx < len(self._frames):
            path = self._frames[self._idx]
            img = cv2.imread(str(path))
            if img is None:
                print(f"Skipping unreadable: {path.name}")
                self._idx += 1
                continue

            self._current_name = path.name
            # Store as RGB for PIL display
            self._orig_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            self._mask = np.zeros((h, w), dtype=np.uint8)
            self._update_status()
            self.root.title(f"Gas Labeler — {self._current_name}")
            return True
        return False

    def _update_status(self):
        labeled_count = len(list(self._masks_dir.glob("*.png")))
        remaining     = len(self._frames) - self._idx
        self._progress_var.set(f"{labeled_count + 1} / {self._total}")
        self._status_var.set(
            f"{self._current_name}  |  Brush: {self.brush_size}  |  {remaining} remaining"
        )

    # ── Display ───────────────────────────────────────────────────────────

    def _build_display_image(self) -> Image.Image:
        """Compose the current image + mask overlay, scaled to the canvas."""
        display = self._orig_image.copy()  # RGB numpy

        # Colour overlay on masked region
        layer = display.copy()
        layer[self._mask > 0] = OVERLAY_COLOR_RGB
        display = cv2.addWeighted(display, 0.55, layer, 0.45, 0)

        # Contour border
        contours, _ = cv2.findContours(
            self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, OVERLAY_COLOR_RGB, 2)

        # Scale to fit canvas
        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)
        h, w = display.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        self._scale    = scale
        self._offset_x = (cw - nw) // 2
        self._offset_y = (ch - nh) // 2

        return Image.fromarray(display).resize((nw, nh), Image.NEAREST)

    def _update_display(self):
        if self._orig_image is None:
            return
        pil_img = self._build_display_image()
        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(
            self._offset_x, self._offset_y,
            anchor=tk.NW, image=self._photo
        )

    def _on_canvas_resize(self, event):
        if self._orig_image is not None:
            if hasattr(self, '_resize_id'):
                self.root.after_cancel(self._resize_id)
            self._resize_id = self.root.after(80, self._update_display)

    # ── Coordinate mapping ────────────────────────────────────────────────

    def _canvas_to_img(self, cx: int, cy: int):
        """Map canvas pixel → image pixel. Returns (ix, iy) or None."""
        if self._scale is None or self._orig_image is None:
            return None
        h, w = self._orig_image.shape[:2]
        ix = int((cx - self._offset_x) / self._scale)
        iy = int((cy - self._offset_y) / self._scale)
        return (max(0, min(w - 1, ix)), max(0, min(h - 1, iy)))

    # ── Mouse events ──────────────────────────────────────────────────────

    def _on_draw_start(self, event):
        self._drawing  = True
        self._erasing  = False
        self._last_pt  = self._canvas_to_img(event.x, event.y)

    def _on_draw_move(self, event):
        if not self._drawing or self._mask is None:
            return
        pt = self._canvas_to_img(event.x, event.y)
        if pt is None:
            return
        if self._last_pt is not None:
            cv2.line(self._mask, self._last_pt, pt, 255, self.brush_size * 2)
            self._update_display()
        self._last_pt = pt

    def _on_draw_stop(self, event):
        self._drawing = False
        self._last_pt = None

    def _on_erase_start(self, event):
        self._erasing  = True
        self._drawing  = False
        self._last_pt  = self._canvas_to_img(event.x, event.y)

    def _on_erase_move(self, event):
        if not self._erasing or self._mask is None:
            return
        pt = self._canvas_to_img(event.x, event.y)
        if pt is None:
            return
        if self._last_pt is not None:
            cv2.line(self._mask, self._last_pt, pt, 0, self.brush_size * 4)
            self._update_display()
        self._last_pt = pt

    def _on_erase_stop(self, event):
        self._erasing = False
        self._last_pt = None

    def _on_scroll(self, event):
        self._adjust_brush(1 if event.delta > 0 else -1)

    # ── Controls ──────────────────────────────────────────────────────────

    def _adjust_brush(self, delta: int):
        self.brush_size = max(1, min(20, self.brush_size + delta))
        if self._current_name:
            self._update_status()

    def _fill_region(self):
        if self._mask is None:
            return
        contours, _ = cv2.findContours(
            self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            filled = np.zeros_like(self._mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
            self._mask = filled
            self._update_display()
            print(f"Filled {len(contours)} region(s)")
        else:
            print("No enclosed region — draw a closed boundary first.")

    def _clear_mask(self):
        if self._mask is not None:
            self._mask = np.zeros_like(self._mask)
            self._update_display()

    # ── Save ──────────────────────────────────────────────────────────────

    def _create_smooth_overlay(self) -> np.ndarray:
        """Return a BGR numpy image with a smooth blended overlay (for saving)."""
        display     = self._orig_image.copy()          # RGB
        smooth_mask = cv2.GaussianBlur(self._mask, (15, 15), 0)
        alpha       = smooth_mask.astype(float) / 255.0
        colored     = np.zeros_like(display)
        colored[:]  = OVERLAY_COLOR_RGB
        for c in range(3):
            display[:, :, c] = (
                (1 - alpha * 0.5) * display[:, :, c] +
                (alpha * 0.5)     * colored[:, :, c]
            )
        return cv2.cvtColor(display.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _save_current(self) -> bool:
        if self._mask is None or np.sum(self._mask) == 0:
            messagebox.showwarning("Nothing Drawn", "Draw a region before saving.")
            return False
        cv2.imwrite(str(self._masks_dir    / self._current_name), self._mask)
        cv2.imwrite(str(self._overlays_dir / self._current_name), self._create_smooth_overlay())
        print(f"Saved: {self._current_name}")
        return True

    def _save_and_next(self):
        if not self._save_current():
            return
        self._idx += 1
        if not self._load_current_frame():
            self._finish()
        else:
            self._update_display()

    def _skip(self):
        print(f"Skipped: {self._current_name}")
        self._idx += 1
        if not self._load_current_frame():
            self._finish()
        else:
            self._update_display()

    def _finish(self):
        masks_done    = len(list(self._masks_dir.glob("*.png")))
        overlays_done = len(list(self._overlays_dir.glob("*.png")))
        messagebox.showinfo(
            "Session Complete",
            f"All images processed!\n\n"
            f"Masks saved    : {masks_done}\n"
            f"Overlays saved : {overlays_done}\n\n"
            f"Location: {self._masks_dir.parent}"
        )
        self._show_home()

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = GasLabelerApp()
    app.run()
