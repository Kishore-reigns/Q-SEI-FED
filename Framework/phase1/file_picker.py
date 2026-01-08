import tkinter as tk
from tkinter import filedialog

def pick_npz_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select IQ Data (.npz)",
        filetypes=[("NPZ files", "*.npz")]
    )
    if not file_path:
        raise FileNotFoundError("No file selected!")
    return file_path
