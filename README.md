# Gas Labeler

Interactive tool to draw and save gas region masks on images.

## Workflow

1. Run the script → a folder picker dialog opens
2. Select the folder containing your images
3. Draw a boundary around the gas region with the left mouse button
4. Press **F** to fill the enclosed region
5. Press **Enter** or **Space** to save and move to the next image

Outputs are saved automatically next to your selected folder:

```
your_folder/
├── frames/       ← images you selected
├── masks/        ← binary masks (auto-created)
└── overlays/     ← smooth color overlays (auto-created)
```

Already-labeled images are skipped automatically when you relaunch.

---

## Controls

| Key / Action         | Function                  |
|----------------------|---------------------------|
| Left Mouse           | Draw boundary             |
| Right Mouse          | Erase                     |
| Scroll / `+` / `-`   | Adjust brush size         |
| `F`                  | Fill enclosed region      |
| `Enter` / `Space`    | Save mask & next image    |
| `N`                  | Skip image (no label)     |
| `C`                  | Clear current mask        |
| `Q` / `Escape`       | Quit                      |

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/taminulislam/label_gas.git
cd label_gas
```

**2. Create a virtual environment**
```bash
python -m venv venv
```

**3. Activate it**

- Windows:
  ```bash
  venv\Scripts\activate
  ```
- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Run**
```bash
python label_gas.py
```
