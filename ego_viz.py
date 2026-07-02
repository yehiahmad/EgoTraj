#!/usr/bin/env python3
import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import TclError
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image, ImageTk
from collections import Counter
import math

# ---------------- Robust file reader ----------------
def read_table_robust(path):
    """
    Reads a tabular file into a pandas DataFrame using various separators and encodings.
    Tries CSV, TSV, and space-delimited with utf-8, latin-1, and cp1252 encodings.
    """
    path = str(path)
    # 1) Try pandas read_csv directly (it guesses separator)
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # 2) Try TSV explicitly
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, sep="\t", encoding=encoding)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    # 3) Try space-delimited
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, delim_whitespace=True, encoding=encoding)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    raise IOError(f"Could not parse {path} with common CSV/TSV/space-delimited settings.")


def detect_time_xy(df):
    """
    Detects time, x, y columns from a DataFrame.

    Time: tries explicit columns like: epoch_time, time, timestamp
    X: columns containing 'x' or 'pos_x', preferring 'x', 'pos_x', 'wx', 'world_x'
    Y: columns containing 'y' or 'z' (for 2D ground-plane: x,z)
    """
    # Normalize columns for matching
    cols = df.columns
    lower = [c.lower() for c in cols]
    colmap = {c.lower(): c for c in cols}

    # 1) Time column
    time_candidates = ["epoch_time", "timestamp", "time", "t"]
    time_col = None
    for cand in time_candidates:
        if cand in colmap:
            time_col = colmap[cand]
            break

    # 2) X, Y detection
    x_col = None
    y_col = None

    # Candidate sets
    cx = ["x", "pos_x", "wx", "world_x"]
    cy = ["z", "pos_z", "wz", "world_z", "y", "pos_y", "wy", "world_y"]

    normalized_cols = {c.lower(): c for c in cols}

    for name in cx:
        if name in normalized_cols:
            x_col = normalized_cols[name]
            break

    for name in cy:
        if name in normalized_cols:
            y_col = normalized_cols[name]
            break

    if x_col is None or y_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col and time_col in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != time_col]
        if len(numeric_cols) >= 2:
            x_col = x_col or numeric_cols[0]
            y_col = y_col or numeric_cols[1]

    if x_col is None or y_col is None:
        raise ValueError("Could not automatically detect two numeric coordinate columns (X and Y/Z).")

    return time_col, x_col, y_col

# ---------------- Core Tkinter Application ----------------
class TkinterCleaner(tk.Tk):
    def __init__(
        self,
        xy,
        headers=("x", "y"),
        time_data=None,
        title="Trajectory Cleaner",
        neighbors=50,
        pad_frac=0.2,
        min_win=None,
        pos_y=None,
        pos_y_name=None,
        speeds=None,
        gaze_dirs=None,
        video_segments=None,
        video_segment_idx=None,
        video_frame_idx=None,
        annotation_text="",
        frame_annotations=None,
        sessions=None,
        current_session=None,
        relaunch_ctx=None,
    ):
        super().__init__()

        self.annotation_text = annotation_text
        self.frame_annotations = frame_annotations
        self.sessions = sessions
        self.current_session = current_session
        self.relaunch_ctx = relaunch_ctx or {}
        self.xy = xy.astype(float)
        self.N = len(self.xy)
        self.headers = headers
        self.keep = np.ones(self.N, dtype=bool)
        self.i = 0

        # Optional extra data (all must match length of xy if provided)
        self.pos_y = pos_y if (pos_y is not None and len(pos_y) == self.N) else None
        self.pos_y_name = pos_y_name
        self.speeds = speeds if (speeds is not None and len(speeds) == self.N) else None
        self.gaze_dirs = gaze_dirs if (gaze_dirs is not None and len(gaze_dirs) == self.N) else None

        # Video sync data
        self.video_segments = video_segments
        self.video_segment_idx = video_segment_idx
        self.video_frame_idx = video_frame_idx
        self.current_photo = None  # keep reference to frame image

        self.time_data = time_data
        if self.time_data is not None:
            self.integer_seconds = self.time_data.astype('datetime64[s]').astype(np.int64)
            self.deleted_per_second = Counter()

        self.neigh = int(max(1, neighbors))
        self.pad_frac = float(pad_frac)

        self.orig_xy = self.xy.copy()
        self.orig_keep = self.keep.copy()
        
        # Compute default min_half from data spread if not specified
        if min_win is None:
            xr = np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0])
            yr = np.nanmax(xy[:, 1]) - np.nanmin(xy[:, 1])
            self.min_half = max(1e-6, 0.005 * max(xr, yr))
        else:
            self.min_half = float(min_win)
        
        self.zoom_factor = 0.15
        self.zoom_scale = 1.0  # 1.0 = default, <1 zoom in, >1 zoom out

        self.view_center = self.xy[self.i].copy()
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._pan_start_view_cx = 0
        self._pan_start_view_cy = 0

        self.title(title)
        self.geometry("1800x1000")
        self.configure(bg="#2b2b2b")
        
        self._create_widgets()
        self._bind_keys()
        
        self.canvas.focus_set()
        self.full_redraw()

    def _relaunch_session(self, session):
        # restart the script with the chosen session, reusing the tested load path
        if session == self.current_session:
            return
        ctx = self.relaunch_ctx
        argv = [sys.executable, ctx.get("script", sys.argv[0]),
                "--h5", ctx["h5"], "--session", session]
        if ctx.get("videos_root"):
            argv += ["--videos-root", ctx["videos_root"]]
        if ctx.get("annotations_json"):
            argv += ["--annotations-json", ctx["annotations_json"]]
        try:
            self.destroy()
        except Exception:
            pass
        os.execv(sys.executable, argv)

    def _create_widgets(self):
        # Top toolbar with a session dropdown (relaunches on selection)
        if self.sessions:
            bar = tk.Frame(self, bg="#1e1e1e")
            bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 0))
            tk.Label(bar, text="Session:", bg="#1e1e1e", fg="#ffffff",
                     font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=(4, 6))
            self.sess_var = tk.StringVar(value=self.current_session or self.sessions[0])
            dd = ttk.Combobox(bar, textvariable=self.sess_var, values=self.sessions,
                              width=26, state="readonly")
            dd.pack(side=tk.LEFT)
            dd.bind("<<ComboboxSelected>>", lambda e: self._relaunch_session(self.sess_var.get()))
            tk.Label(bar, text="(switching reloads the viewer)", bg="#1e1e1e",
                     fg="#888888", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)

        main_frame = tk.Frame(self, bg="#2b2b2b")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # LEFT: trajectory canvas + bottom info
        canvas_frame = tk.Frame(main_frame, bg="#2b2b2b")
        canvas_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, bg="#0e0f12", highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        bottom_bar = tk.Frame(canvas_frame, bg="#2b2b2b")
        bottom_bar.pack(fill=tk.X, pady=(5, 0))
        
        self.info_var = tk.StringVar()
        info_label = tk.Label(
            bottom_bar,
            textvariable=self.info_var,
            bg="#2b2b2b",
            fg="#ffd34d",
            font=("Segoe UI", 13),
            anchor='w',
        )
        info_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        jump_frame = tk.Frame(bottom_bar, bg="#2b2b2b")
        jump_frame.pack(side=tk.RIGHT)

        go_label = tk.Label(
            jump_frame,
            text="Go to Index:",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Segoe UI", 12),
        )
        go_label.pack(side=tk.LEFT, padx=(10, 5))

        self.jump_entry = tk.Entry(
            jump_frame,
            width=12,
            bg="#ffffff",
            fg="#000000",
            font=("Segoe UI", 12),
        )
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", self._jump_to_index)
        self.jump_entry.bind("<KP_Enter>", self._jump_to_index)

        # RIGHT: minimap (full path), frame scene, annotation, instructions
        right_panel = tk.Frame(main_frame, bg="#2b2b2b")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Full path minimap (fixed small height so the annotation panel has room)
        minimap_frame = tk.Frame(right_panel, bg="#2b2b2b", height=170)
        minimap_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=(0, 5))
        minimap_frame.pack_propagate(False)

        tk.Label(
            minimap_frame,
            text="Full Path Trajectory",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Segoe UI", 13, "bold"),
        ).pack()

        self.minimap = tk.Canvas(
            minimap_frame,
            bg="#0e0f12",
            highlightthickness=0,
            height=140,
        )
        self.minimap.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Frame scene
        video_frame = tk.Frame(right_panel, bg="#2b2b2b")
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))

        tk.Label(
            video_frame,
            text="Frame View",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Segoe UI", 13, "bold"),
        ).pack()

        self.video_canvas = tk.Canvas(
            video_frame,
            bg="#000000",
            highlightthickness=0,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Scene annotation panel
        ann_frame = tk.Frame(right_panel, bg="#252525", padx=15, pady=10)
        ann_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))

        tk.Label(
            ann_frame,
            text="Scene Annotation",
            bg="#252525",
            fg="#00ffff",
            font=("Segoe UI", 13, "bold"),
            anchor='nw',
        ).pack(anchor='w')

        self.ann_text_label = tk.Label(
            ann_frame,
            text=self.annotation_text or "No annotation loaded",
            bg="#252525",
            fg="#ffd34d" if self.annotation_text else "#666666",
            font=("Segoe UI", 12),
            anchor='nw',
            justify=tk.LEFT,
        )
        self.ann_text_label.pack(anchor='w', fill=tk.BOTH, expand=True, pady=(5, 0))

        # Dynamic wraplength on resize
        def _update_wraplength(event):
            self.ann_text_label.config(wraplength=max(event.width - 30, 50))
        ann_frame.bind("<Configure>", _update_wraplength)

        # Lower-right: instructions panel
        self._create_instructions_panel(right_panel)

    def _create_instructions_panel(self, parent):
        dark_grey = "#252525"
        bright_yellow = "#ffff00"
        
        instr_frame = tk.Frame(parent, bg=dark_grey, padx=20, pady=20)
        instr_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, ipadx=10)

        def add_instruction(parent, key, desc):
            row = tk.Frame(parent, bg=dark_grey)
            key_label = tk.Label(row, text=f"[{key}]", bg=dark_grey, fg=bright_yellow, font=("Consolas", 12, "bold"), width=18, anchor='w')
            desc_label = tk.Label(row, text=desc, bg=dark_grey, fg="#ffffff", font=("Segoe UI", 12), anchor='w')
            key_label.pack(side=tk.LEFT)
            desc_label.pack(side=tk.LEFT)
            row.pack(anchor='w', pady=3)

        def add_header(parent, text):
            header = tk.Label(parent, text=text, bg=dark_grey, fg="#00ffff", font=("Segoe UI", 14, "bold"), anchor='w')
            header.pack(anchor='w', pady=(15, 8))

        add_header(instr_frame, "Navigation")
        add_instruction(instr_frame, "Left / Right", "Move to previous / next point")
        add_instruction(instr_frame, "Up / Down", "Jump by 10 points")
        add_instruction(instr_frame, "PageUp / PageDown", "Jump by 100 points")
        add_instruction(instr_frame, "Home / End", "Jump to start / end")

        add_header(instr_frame, "Editing")
        add_instruction(instr_frame, "D", "Delete current point")
        add_instruction(instr_frame, "K", "Keep (undelete) current point")
        add_instruction(instr_frame, "Enter", "Apply deletions permanently")
        
        add_header(instr_frame, "View Control")
        add_instruction(instr_frame, "Mouse Wheel", "Zoom in / out around the current view")
        add_instruction(instr_frame, "Right Mouse + Drag", "Pan the view")
        
        add_header(instr_frame, "Other")
        add_instruction(instr_frame, "S", "Save cleaned data")
        add_instruction(instr_frame, "H / ?", "Show help window")
        add_instruction(instr_frame, "Q or Esc", "Quit")

    def _show_help_dialog(self):
        """Show keyboard shortcuts help dialog (learned from enhanced version)."""
        help_win = tk.Toplevel(self)
        help_win.title("Keyboard Shortcuts")
        help_win.geometry("850x900")
        help_win.configure(bg="#2b2b2b")
        
        # Make it modal
        help_win.transient(self)
        help_win.grab_set()
        
        # Create scrollable frame
        canvas = tk.Canvas(help_win, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(help_win, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2b2b2b")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title = tk.Label(
            scrollable_frame,
            text="⌨️  Keyboard Shortcuts",
            bg="#2b2b2b",
            fg="#00ffff",
            font=("Segoe UI", 18, "bold"),
            pady=20
        )
        title.pack()
        
        sections = [
            ("Navigation", [
                ("←  / →", "Previous / Next point"),
                ("↑  / ↓", "Jump by 10 points"),
                ("PgUp / PgDn", "Jump by 100 points"),
                ("Home / End", "Jump to start / end"),
            ]),
            ("Editing", [
                ("D", "Delete current point"),
                ("K", "Keep (undelete) current point"),
                ("Enter", "Apply deletions permanently"),
            ]),
            ("View Control", [
                ("Mouse Wheel", "Zoom in / out"),
                ("Right Click + Drag", "Pan view"),
            ]),
            ("Other", [
                ("S", "Save cleaned data"),
                ("H or ?", "Show this help"),
                ("Q or Esc", "Quit"),
            ]),
        ]
        
        for section_name, shortcuts in sections:
            # Section header
            header = tk.Label(
                scrollable_frame,
                text=section_name,
                bg="#2b2b2b",
                fg="#ffd34d",
                font=("Segoe UI", 14, "bold"),
                anchor="w",
                pady=10
            )
            header.pack(fill=tk.X, padx=30)
            
            # Shortcuts
            for key, desc in shortcuts:
                row = tk.Frame(scrollable_frame, bg="#2b2b2b")
                row.pack(fill=tk.X, padx=50, pady=2)
                
                key_label = tk.Label(
                    row,
                    text=key,
                    bg="#252525",
                    fg="#ffff00",
                    font=("Consolas", 13, "bold"),
                    width=18,
                    anchor="w",
                    padx=10,
                    pady=5
                )
                key_label.pack(side=tk.LEFT, padx=(0, 10))
                
                desc_label = tk.Label(
                    row,
                    text=desc,
                    bg="#2b2b2b",
                    fg="#ffffff",
                    font=("Segoe UI", 12),
                    anchor="w"
                )
                desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Close button
        close_btn = tk.Button(
            scrollable_frame,
            text="Close",
            command=help_win.destroy,
            bg="#3f51b5",
            fg="#ffffff",
            font=("Segoe UI", 13, "bold"),
            padx=30,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        close_btn.pack(pady=20)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Center the window
        help_win.update_idletasks()
        x = (help_win.winfo_screenwidth() // 2) - (help_win.winfo_width() // 2)
        y = (help_win.winfo_screenheight() // 2) - (help_win.winfo_height() // 2)
        help_win.geometry(f"+{x}+{y}")

    def _bind_keys(self):
        self.canvas.bind("<Left>", lambda e: self._move_index(-1))
        self.canvas.bind("<Right>", lambda e: self._move_index(1))
        self.canvas.bind("<Up>", lambda e: self._move_index(-10))
        self.canvas.bind("<Down>", lambda e: self._move_index(10))
        self.canvas.bind("<Prior>", lambda e: self._move_index(-100))  # PageUp
        self.canvas.bind("<Next>", lambda e: self._move_index(100))    # PageDown
        self.canvas.bind("<Home>", lambda e: self._jump_to(0))
        self.canvas.bind("<End>", lambda e: self._jump_to(self.N - 1))

        self.canvas.bind("<d>", lambda e: self._delete_and_advance())
        self.canvas.bind("<D>", lambda e: self._delete_and_advance())
        self.canvas.bind("<k>", lambda e: self._undelete_and_advance())
        self.canvas.bind("<K>", lambda e: self._undelete_and_advance())

        self.canvas.bind("<Return>", lambda e: self._apply_and_redraw())
        self.canvas.bind("<space>", lambda e: self._downsample_to_30hz())

        self.canvas.bind("<q>", lambda e: self.quit_app())
        self.canvas.bind("<Q>", lambda e: self.quit_app())
        self.canvas.bind("<Escape>", lambda e: self.quit_app())

        # Help bindings (like enhanced version)
        self.canvas.bind("<question>", lambda e: self._show_help_dialog())
        self.canvas.bind("<Shift-slash>", lambda e: self._show_help_dialog())  # Shift+/ = ?
        self.canvas.bind("<h>", lambda e: self._show_help_dialog())
        self.canvas.bind("<H>", lambda e: self._show_help_dialog())

        self.canvas.bind("<MouseWheel>", self._on_zoom)          # Windows
        self.canvas.bind("<Button-4>", self._on_zoom)            # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_zoom)            # Linux scroll down

        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._perform_pan)

        self.bind("<MouseWheel>", self._on_zoom)
        self.bind("<Button-4>", self._on_zoom)
        self.bind("<Button-5>", self._on_zoom)

    # ---------- Geometry helpers ----------
    def world_to_canvas(self, x, y):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return 0, 0
        px = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0] + 1e-12)
        py = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0] + 1e-12)
        sx = 20 + px * (w - 40)
        sy = h - (20 + py * (h - 40))
        return sx, sy

    def canvas_to_world(self, sx, sy):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        px = (sx - 20) / max(w - 40, 1e-12)
        py = (h - sy - 20) / max(h - 40, 1e-12)
        x = self.xlim[0] + px * (self.xlim[1] - self.xlim[0])
        y = self.ylim[0] + py * (self.ylim[1] - self.ylim[0])
        return x, y

    # ---------- Grid drawing ----------
    def _draw_grid(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        bg_col = "#0e0f12"
        grid_col = "#222222"
        axis_col = "#888888"
        label_col = "#aaaaaa"
        self.canvas.create_rectangle(0, 0, w, h, fill=bg_col, outline=bg_col)

        def nice_step(vmin, vmax, n=10):
            if vmin == vmax:
                return 1.0
            raw = (vmax - vmin) / n
            if raw <= 0:
                raw = abs(raw) + 1e-6
            mag = 10 ** math.floor(math.log10(raw))
            norm = raw / mag
            if norm < 1.5:
                step = 1.0
            elif norm < 3:
                step = 2.0
            elif norm < 7:
                step = 5.0
            else:
                step = 10.0
            return step * mag

        x_min, x_max = self.xlim
        y_min, y_max = self.ylim

        xs = nice_step(x_min, x_max, 8)
        x0 = xs * math.floor(x_min / xs)
        x = x0
        while x <= x_max + 1e-12:
            sx, _ = self.world_to_canvas(x, 0)
            self.canvas.create_line(sx, 0, sx, h, fill=grid_col, width=1)
            if abs(x) < 1e-12:
                self.canvas.create_line(sx, 0, sx, h, fill=axis_col, width=2)
            self.canvas.create_text(sx, h-5, text=f"{x:.2f}", fill=label_col, font=("Consolas", 11))
            x += xs

        ys = nice_step(y_min, y_max, 8)
        y0 = ys * math.floor(y_min / ys)
        y = y0
        while y <= y_max + 1e-12:
            _, sy = self.world_to_canvas(0, y)
            self.canvas.create_line(0, sy, w, sy, fill=grid_col, width=1)
            if abs(y) < 1e-12:
                self.canvas.create_line(0, sy, w, sy, fill=axis_col, width=2)
            self.canvas.create_text(6, sy - 2, text=f"{y:.2f}", anchor='w', fill=label_col, font=("Consolas", 11))
            y += ys

    # ---------- Drawing ----------
    def full_redraw(self):
        self.canvas.delete("all")
        self._update_view_limits()

        # draw grid first
        self._draw_grid()
        
        # Determine visible indices
        visible_indices = np.where(
            (self.xy[:, 0] >= self.xlim[0]) &
            (self.xy[:, 0] <= self.xlim[1]) &
            (self.xy[:, 1] >= self.ylim[0]) &
            (self.xy[:, 1] <= self.ylim[1])
        )[0]

        # Draw trajectory polyline
        screen_coords = [self.world_to_canvas(p[0], p[1]) for p in self.xy]
        if len(screen_coords) >= 2:
            self.canvas.create_line(screen_coords, fill="#ad5cad", width=4.0)

        # Draw points
        for idx in visible_indices:
            p = self.xy[idx]
            x, y = self.world_to_canvas(p[0], p[1])
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="#00e500", outline="")
            if idx % 1000 == 0:
                self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="#3f51b5", outline="")
        
        # Draw deleted points (X mark)
        deleted_indices_in_view = np.intersect1d(visible_indices, np.where(~self.keep)[0])
        for idx in deleted_indices_in_view:
            p = self.xy[idx]
            x, y = self.world_to_canvas(p[0], p[1])
            self.canvas.create_line(x-7, y-7, x+7, y+7, fill="#ff0000", width=1.5)
            self.canvas.create_line(x-7, y+7, x+7, y-7, fill="#ff0000", width=1.5)

        # Highlight current point
        cx, cy = self.world_to_canvas(self.xy[self.i, 0], self.xy[self.i, 1])
        r = 10
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#00ffff", width=2.0)

        point_label_text = f"{self.i + 1}/{self.N}"
        self.canvas.create_text(
            cx + r + 5,
            cy,
            text=point_label_text,
            anchor='w',
            fill="#ffd34d",
            font=("Segoe UI", 12, "bold"),
        )

        # Draw gaze direction arrow, if available
        if self.gaze_dirs is not None:
            gx, gz = self.gaze_dirs[self.i]
            px, pz = self.xy[self.i]
            gaze_length = 0.7  # meters

            end_x = px + gx * gaze_length
            end_z = pz + gz * gaze_length

            sx1, sy1 = self.world_to_canvas(px, pz)
            sx2, sy2 = self.world_to_canvas(end_x, end_z)

            outline_width = 5

            # White outline arrow
            self.canvas.create_line(
                sx1, sy1, sx2, sy2,
                fill="#ffffff",
                width=outline_width,
                arrow=tk.LAST,
                arrowshape=(12, 14, 9),
            )

        # Time overlay (top-left)
        if self.time_data is not None:
            current_dt = pd.to_datetime(self.time_data[self.i])
            ttxt = current_dt.strftime('%H:%M:%S.%f')[:-3]  # ms precision
            self.canvas.create_text(
                12,
                14,
                anchor='w',
                text=f"t = {ttxt}",
                fill="#00ffff",
                font=("Consolas", 14, "bold"),
            )

        # Bottom info bar
        self._update_hud()

        # Upper-right frame view
        self._draw_video_frame()

        # Full-path minimap on the right
        self._draw_minimap()

    def _draw_minimap(self):
        if not hasattr(self, 'minimap'):
            return
        
        self.minimap.delete("all")
        
        w = self.minimap.winfo_width()
        h = self.minimap.winfo_height()
        if w < 10 or h < 10:
            return
        
        # Calculate bounds from all points
        x_min = np.nanmin(self.xy[:, 0])
        x_max = np.nanmax(self.xy[:, 0])
        y_min = np.nanmin(self.xy[:, 1])
        y_max = np.nanmax(self.xy[:, 1])
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range < 1e-6 or y_range < 1e-6:
            return
        
        pad = 10
        
        def map_to_canvas(x, y):
            px = (x - x_min) / x_range
            py = (y - y_min) / y_range
            sx = pad + px * (w - 2 * pad)
            sy = h - (pad + py * (h - 2 * pad))
            return sx, sy
        
        # Draw trajectory as a thin line (subsample for speed)
        points = []
        step = max(1, self.N // 500)
        for i in range(0, self.N, step):
            sx, sy = map_to_canvas(self.xy[i, 0], self.xy[i, 1])
            points.extend([sx, sy])
        
        if len(points) >= 4:
            self.minimap.create_line(points, fill="#666666", width=1)

        vx_min, vx_max = self.xlim
        vy_min, vy_max = self.ylim

        # Map the 4 corners of the current view to minimap coords
        corners = [
            map_to_canvas(vx_min, vy_min),
            map_to_canvas(vx_max, vy_min),
            map_to_canvas(vx_max, vy_max),
            map_to_canvas(vx_min, vy_max),
        ]

        self.minimap.create_polygon(
            corners[0][0], corners[0][1],
            corners[1][0], corners[1][1],
            corners[2][0], corners[2][1],
            corners[3][0], corners[3][1],
            outline="#00ffff",   # cyan box
            fill="",
            width=2,
        )

        # Draw current point
        cx, cy = map_to_canvas(self.xy[self.i, 0], self.xy[self.i, 1])
        self.minimap.create_oval(cx-3, cy-3, cx+3, cy+3, fill="#ffd34d", outline="")

    def _draw_video_frame(self):
        """Draw synchronized video frame (if available) in the upper-right canvas."""
        # update the per-frame annotation on every frame change (independent of video)
        if getattr(self, "frame_annotations", None) is not None and hasattr(self, "ann_text_label"):
            txt = self.frame_annotations[self.i] if self.i < len(self.frame_annotations) else ""
            txt = str(txt) if txt else ""
            self.ann_text_label.config(
                text=txt if txt else "(no annotation for this second)",
                fg="#ffd34d" if txt else "#666666",
            )

        # If video canvas does not exist, nothing to do
        if not hasattr(self, "video_canvas"):
            return

        self.video_canvas.delete("all")

        # No video loaded at all
        if (
            self.video_segments is None
            or self.video_segment_idx is None
            or self.video_frame_idx is None
        ):
            w = self.video_canvas.winfo_width()
            h = self.video_canvas.winfo_height()
            if w < 10 or h < 10:
                return
            self.video_canvas.create_text(
                w / 2,
                h / 2,
                text="No video loaded",
                fill="#888888",
                font=("Segoe UI", 13),
            )
            return

        seg_idx = self.video_segment_idx[self.i]
        frm_idx = self.video_frame_idx[self.i]

        # Handle NaNs or invalid entries
        if (
            seg_idx is None
            or frm_idx is None
            or (isinstance(seg_idx, float) and np.isnan(seg_idx))
            or (isinstance(frm_idx, float) and np.isnan(frm_idx))
        ):
            w = self.video_canvas.winfo_width()
            h = self.video_canvas.winfo_height()
            if w < 10 or h < 10:
                return
            self.video_canvas.create_text(
                w / 2,
                h / 2,
                text="No video for this point",
                fill="#888888",
                font=("Segoe UI", 13),
            )
            return

        try:
            seg_idx = int(seg_idx)
            frm_idx = int(frm_idx)
        except Exception:
            return

        if seg_idx < 0 or self.video_segments is None or seg_idx >= len(self.video_segments):
            w = self.video_canvas.winfo_width()
            h = self.video_canvas.winfo_height()
            if w < 10 or h < 10:
                return
            self.video_canvas.create_text(
                w / 2,
                h / 2,
                text="Invalid segment index",
                fill="#ff4444",
                font=("Segoe UI", 13),
            )
            return

        segment = self.video_segments[seg_idx]
        cap = segment["capture"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            w = self.video_canvas.winfo_width()
            h = self.video_canvas.winfo_height()
            if w < 10 or h < 10:
                return
            self.video_canvas.create_text(
                w / 2,
                h / 2,
                text="Failed to read frame",
                fill="#ff4444",
                font=("Segoe UI", 13),
            )
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            return

        h_img, w_img = frame_rgb.shape[:2]
        scale = min(canvas_width / w_img, canvas_height / h_img)

        new_w = int(w_img * scale)
        new_h = int(h_img * scale)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # Convert to PIL Image then to Tkinter PhotoImage
        img = Image.fromarray(frame_resized)
        self.current_photo = ImageTk.PhotoImage(img)

        x_off = (canvas_width - new_w) // 2
        y_off = (canvas_height - new_h) // 2

        self.video_canvas.create_image(x_off, y_off, image=self.current_photo, anchor=tk.NW)

        # Overlay text with segment/frame indices
        self.video_canvas.create_text(
            10,
            10,
            anchor=tk.NW,
            text=f"Frame {frm_idx}",
            fill="#ffff00",
            font=("Segoe UI", 12, "bold"),
        )

    def _update_view_limits(self):
        cx, cy = self.view_center[0], self.view_center[1]
        lo = max(0, self.i - self.neigh)
        hi = min(self.N - 1, self.i + self.neigh)
        pts = self.xy[lo:hi+1]
        xmin, xmax = np.nanmin(pts[:, 0]), np.nanmax(pts[:, 0])
        ymin, ymax = np.nanmin(pts[:, 1]), np.nanmax(pts[:, 1])

        xr = xmax - xmin
        yr = ymax - ymin
        xr = max(xr, self.min_half)
        yr = max(yr, self.min_half)

        base_half_span = 0.5 * max(xr, yr)
        base_half_span *= (1.0 + self.pad_frac)

        # Apply zoom scale (zoom_scale < 1 => zoom in, >1 => zoom out)
        half_span = base_half_span * self.zoom_scale

        self.xlim = (cx - half_span, cx + half_span)
        self.ylim = (cy - half_span, cy + half_span)

    def _update_hud(self):
        kept = int(self.keep.sum())
        base_info = f"Index: {self.i+1}/{self.N}  |  Kept: {kept}  |  Deleted: {self.N - kept}"

        # Position info
        x = self.xy[self.i, 0]
        z = self.xy[self.i, 1]
        if self.pos_y is not None:
            pos_str = f"Pos: ({x:.2f}, {self.pos_y[self.i]:.2f}, {z:.2f})"
        else:
            pos_str = f"Pos: ({x:.2f}, {z:.2f})"

        # Speed info
        if self.speeds is not None:
            speed_str = f"  |  Speed: {self.speeds[self.i]:.2f} m/s"
        else:
            speed_str = ""

        if self.time_data is not None:
            current_dt = pd.to_datetime(self.time_data[self.i])
            time_str = current_dt.strftime('%H:%M:%S') + f".{current_dt.microsecond // 1000:03d}"
            
            current_second = self.integer_seconds[self.i]
            deleted_count = self.deleted_per_second.get(current_second, 0)
            
            time_info = f"Time: {time_str}  |  Deleted in this sec: {deleted_count}"
            self.info_var.set(f"{time_info}  |  {pos_str}{speed_str}  |  {base_info}")
        else:
            self.info_var.set(f"{pos_str}{speed_str}  |  {base_info}")

    # ---------- Navigation & editing ----------
    def _move_index(self, di):
        new_i = int(np.clip(self.i + di, 0, self.N - 1))
        if new_i != self.i:
            self.i = new_i
            self.view_center = self.xy[self.i].copy()
            self.full_redraw()

    def _jump_to(self, idx):
        idx = int(np.clip(idx, 0, self.N - 1))
        self.i = idx
        self.view_center = self.xy[self.i].copy()
        self.full_redraw()

    def _jump_to_index(self, event=None):
        input_text = self.jump_entry.get().strip()  # Get directly from widget!
        try:
            if not input_text:
                return  # Empty, just ignore
            target_idx = int(input_text) - 1
            if 0 <= target_idx < self.N:
                self.i = target_idx
                self.view_center = self.xy[self.i].copy()
                self.jump_entry.delete(0, tk.END)
                self.canvas.focus_set()
                self.full_redraw()
            else:
                messagebox.showwarning("Invalid Index", f"Index must be between 1 and {self.N}")
                self.jump_entry.delete(0, tk.END)
                self.canvas.focus_set()
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid Input", f"Please enter a valid number (got: '{input_text}')")
            self.jump_entry.delete(0, tk.END)
            self.canvas.focus_set()

    def _set_keep(self, keep_flag):
        if self.keep[self.i] == keep_flag:
            return False

        if self.time_data is not None:
            current_second = self.integer_seconds[self.i]
            if keep_flag:
                self.deleted_per_second[current_second] = self.deleted_per_second.get(current_second, 1) - 1
                if self.deleted_per_second[current_second] <= 0:
                    self.deleted_per_second.pop(current_second, None)
            else:
                self.deleted_per_second[current_second] = self.deleted_per_second.get(current_second, 0) + 1
        
        self.keep[self.i] = keep_flag
        return True

    def _delete_and_advance(self):
        if self._set_keep(False):
            if self.i < self.N - 1:
                self.i += 1
                self.view_center = self.xy[self.i].copy()
            self.full_redraw()

    def _undelete_and_advance(self):
        if self._set_keep(True):
            if self.i < self.N - 1:
                self.i += 1
                self.view_center = self.xy[self.i].copy()
            self.full_redraw()

    # Apply deletions permanently and update path/arrays
    def _apply_and_redraw(self):
        if np.all(self.keep):
            return

        keep_mask = self.keep.copy()

        # Core coordinates
        self.xy = self.xy[keep_mask]

        # Time
        if self.time_data is not None:
            self.time_data = self.time_data[keep_mask]
            self.integer_seconds = self.time_data.astype('datetime64[s]').astype(np.int64)
            self.deleted_per_second = Counter()

        # Extra arrays stay in sync
        if self.pos_y is not None:
            self.pos_y = self.pos_y[keep_mask]
        if self.speeds is not None:
            self.speeds = self.speeds[keep_mask]
        if self.gaze_dirs is not None:
            self.gaze_dirs = self.gaze_dirs[keep_mask]
        if self.video_segment_idx is not None:
            self.video_segment_idx = self.video_segment_idx[keep_mask]
        if self.video_frame_idx is not None:
            self.video_frame_idx = self.video_frame_idx[keep_mask]
        if self.frame_annotations is not None:
            self.frame_annotations = self.frame_annotations[keep_mask]

        self.N = len(self.xy)
        self.keep = np.ones(self.N, dtype=bool)
        self.i = int(np.clip(self.i, 0, self.N - 1))
        self.view_center = self.xy[self.i].copy()
        self.full_redraw()

    # Downsample to ~30 Hz using nearest timestamps (no interpolation)
    def _downsample_to_30hz(self):
        if self.time_data is None:
            messagebox.showwarning(
                "No time column",
                "To downsample to 30 Hz, your file must include a time column (epoch_time/timestamp/time).",
            )
            return
        if len(self.time_data) < 2:
            return

        t = self.time_data.astype('datetime64[ns]').astype('int64') / 1e9  # seconds float
        dt = np.diff(t)
        med_dt = float(np.median(dt))
        if med_dt <= 0 or not np.isfinite(med_dt):
            messagebox.showwarning("Invalid time", "Timestamps are not strictly increasing; cannot resample.")
            return
        orig_hz = 1.0 / med_dt

        target_hz = 30.0
        if orig_hz <= target_hz + 1e-6:
            messagebox.showinfo("Already ≤ 30 Hz", f"Detected sampling rate ≈ {orig_hz:.2f} Hz — no downsampling applied.")
            return

        step = 1.0 / target_hz
        t0 = t[0]
        tN = t[-1]
        new_times = np.arange(t0, tN + step * 0.5, step)

        idxs = []
        last_idx = -1
        n = len(t)
        for tt in new_times:
            j = int(np.searchsorted(t, tt))
            if j == 0:
                cand = 0
            elif j >= n:
                cand = n - 1
            else:
                if abs(t[j] - tt) < abs(tt - t[j-1]):
                    cand = j
                else:
                    cand = j - 1
            if cand > last_idx:
                idxs.append(cand)
                last_idx = cand

        if len(idxs) < 2:
            messagebox.showwarning("Downsample failed", "Could not compute a valid 30 Hz subset.")
            return

        idxs = np.array(idxs, dtype=int)

        # Apply index subset to all arrays
        self.xy = self.xy[idxs]
        if self.time_data is not None:
            self.time_data = self.time_data[idxs]
            self.integer_seconds = self.time_data.astype('datetime64[s]').astype(np.int64)
            self.deleted_per_second = Counter()
        if self.pos_y is not None:
            self.pos_y = self.pos_y[idxs]
        if self.speeds is not None:
            self.speeds = self.speeds[idxs]
        if self.gaze_dirs is not None:
            self.gaze_dirs = self.gaze_dirs[idxs]
        if self.video_segment_idx is not None:
            self.video_segment_idx = self.video_segment_idx[idxs]
        if self.video_frame_idx is not None:
            self.video_frame_idx = self.video_frame_idx[idxs]
        if self.frame_annotations is not None:
            self.frame_annotations = self.frame_annotations[idxs]

        self.N = len(self.xy)
        self.keep = np.ones(self.N, dtype=bool)
        self.i = min(self.i, self.N - 1)
        self.view_center = self.xy[self.i].copy()
        self.full_redraw()
        messagebox.showinfo("Downsampled", f"Original rate ≈ {orig_hz:.2f} Hz → 30 Hz.\nPoints kept: {self.N}")

    # ---------- Zoom & pan ----------
    def _on_zoom(self, event):
        # Scroll up (or Button-4 on Linux) => zoom in
        if event.delta > 0 or getattr(event, "num", None) == 4:
            self.zoom_scale *= (1.0 - self.zoom_factor)
        else:
            self.zoom_scale *= (1.0 + self.zoom_factor)

        # Keep zoom in a sane range
        self.zoom_scale = float(np.clip(self.zoom_scale, 0.05, 20.0))

        self.full_redraw()

    def _start_pan(self, event):
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._pan_start_view_cx, self._pan_start_view_cy = self.view_center

    def _perform_pan(self, event):
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        scale_x = (self.xlim[1] - self.xlim[0]) / (w - 40 + 1e-12)
        scale_y = (self.ylim[1] - self.ylim[0]) / (h - 40 + 1e-12)

        self.view_center = np.array([
            self._pan_start_view_cx - dx * scale_x,
            self._pan_start_view_cy + dy * scale_y
        ])
        self.full_redraw()

    # ---------- Save & quit ----------
    def _save(self):
        base, ext = os.path.splitext(self.original_path)
        out_path = base + "_cleaned" + ext
        try:
            df = self.original_df.loc[self.keep_original_mask].copy()
            df.to_csv(out_path, index=False)
            messagebox.showinfo("Saved", f"Saved {len(df)} points to:\n{os.path.basename(out_path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def quit_app(self):
        if messagebox.askyesno("Save before quit?", "Save cleaned data before exiting?"):
            self._save()
        self.destroy()

# --- FIXED: Data loading logic is now robust ---

def load_video_segments(session_folder: Path):
    """Load all video_*_part*.mp4 segments from a session folder."""
    print("Loading video segments...")

    video_segments = []
    video_files = sorted(session_folder.glob("video_*_part*.mp4"))
    for i, video_path in enumerate(video_files):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠ Warning: Could not open {video_path.name}")
            continue

        segment = {
            "index": i,
            "path": video_path,
            "capture": cap,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        video_segments.append(segment)
        print(f"  ✓ Loaded segment {i}: {video_path.name}")

    if not video_segments:
        print("No video segments found in session folder.")
    return video_segments

def load_from_h5(h5_path, session):
    # build the same df shape unified_data_loader produces, straight from the built h5
    import h5py
    f = h5py.File(h5_path, "r")
    if session not in f:
        keys = list(f.keys())
        f.close()
        raise KeyError(f"session '{session}' not in h5. available: {keys[:5]} ... ({len(keys)} total)")
    g = f[session]
    pos = g["pose/position"][:]
    rot = g["pose/rotation"][:]
    vel = g["pose/velocity"][:]
    angv = g["pose/angular_velocity"][:]
    ts = g["pose/timestamp"][:]
    gorg = g["gaze/origin"][:]
    gdir = g["gaze/direction"][:]
    seg = g["video/segment"][:] if "video/segment" in g else None
    frm = g["video/frame"][:] if "video/frame" in g else None
    hv = g["video/has_video"][:] if "video/has_video" in g else None
    f.close()

    df = pd.DataFrame({
        "timestamp": ts,
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        # quest pro order qx,qy,qz,qw; kept for save only, not used by the 2d view
        "qx": rot[:, 0], "qy": rot[:, 1], "qz": rot[:, 2], "qw": rot[:, 3],
        "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
        "angv_x": angv[:, 0], "angv_y": angv[:, 1], "angv_z": angv[:, 2],
        "gaze_origin_wx": gorg[:, 0], "gaze_origin_wy": gorg[:, 1], "gaze_origin_wz": gorg[:, 2],
        "gaze_dir_wx": gdir[:, 0], "gaze_dir_wy": gdir[:, 1], "gaze_dir_wz": gdir[:, 2],
    })
    # mark unmatched frames as nan so the viz treats them as no-frame, same as the csv path
    if seg is not None and frm is not None:
        segf = seg.astype("float64")
        frmf = frm.astype("float64")
        if hv is not None:
            segf[~hv] = np.nan
            frmf[~hv] = np.nan
        df["video_segment"] = segf
        df["video_frame"] = frmf
    return df


def main():
    ap = argparse.ArgumentParser(description="Ultra-fast cursor-based point remover with local zoom")
    ap.add_argument("file", nargs="?", help="CSV/TXT/TSV with X,Y (or X,Z) and optional epoch_time")
    ap.add_argument("--h5", type=str, default=None, help="path to egotraj_dataset.h5 (use with --session)")
    ap.add_argument("--session", type=str, default=None, help="session key inside the h5, e.g. 20251020_163423")
    ap.add_argument("--videos", type=str, default=None, help="folder with that session's video_*_part*.mp4 (optional)")
    ap.add_argument("--videos-root", type=str, default=None, help="parent folder of per-session video subfolders; enables the session dropdown")
    ap.add_argument("--neighbors", type=int, default=50, help="Points on each side to define the zoom window (default: 50)")
    ap.add_argument("--pad", type=float, default=0.2, help="Padding around local bbox as a fraction (default: 0.2)")
    ap.add_argument("--minwin", type=float, default=None, help="Minimum half-window size in data units (default: auto)")
    ap.add_argument("--annotations", type=str, default=None, help="Path to scene annotation .txt file")
    ap.add_argument("--annotations-json", type=str, default=None, help="Path to annotations.json (per-frame, keyed by pose_idx); shows the annotation for the current second")
    args = ap.parse_args()

    if args.h5:
        if not args.session:
            print("ERROR: --session is required with --h5"); sys.exit(1)
        df = load_from_h5(args.h5, args.session)
        # resolve this session's video folder: explicit --videos wins, else <videos-root>/<session>
        if args.videos:
            vid_dir = args.videos
        elif args.videos_root:
            vid_dir = str(Path(args.videos_root) / args.session)
        else:
            vid_dir = str(Path(args.h5).parent)
        path = str(Path(vid_dir) / f"{args.session}_h5.csv")
    else:
        path = args.file
        if not path:
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(title="Choose a CSV/TXT/TSV file", filetypes=[("Data files", "*.csv *.txt *.tsv"), ("All files", "*.*")])
            if not path: sys.exit(0)

    try:
        if not args.h5:
            df = read_table_robust(path)
        time_col, xcol, ycol = detect_time_xy(df)

        print(f"DEBUG: Detected columns - Time: {time_col}, X: {xcol}, Y: {ycol}")  # Debug
        print(f"DEBUG: Available columns: {df.columns.tolist()}")  # Debug
        
        if time_col is None:
            print("INFO: Time column (e.g., 'epoch_time') not found. Proceeding in index-only mode.")

        # Define all columns that must have valid data
        required_cols = [xcol, ycol]
        if time_col:
            required_cols.append(time_col)

        # Drop any row that has a NaN in any of the essential columns
        df.dropna(subset=required_cols, inplace=True)
        
        if df.empty:
            messagebox.showerror("Error", "No valid numeric data found in the file after cleaning.")
            sys.exit(1)
            
        # Create time_data array AFTER cleaning the dataframe to ensure they match
        time_data = None
        if time_col:
            # Detect if epoch is in seconds or milliseconds
            sample = pd.to_numeric(df[time_col].iloc[:100], errors='coerce').dropna()
            guess = float(sample.median()) if not sample.empty else float(df[time_col].iloc[0])
            unit_used = 'ms' if guess > 1e11 else 's'
            time_data = pd.to_datetime(df[time_col], unit=unit_used, errors='coerce').to_numpy()
            print(f"INFO: Detected epoch unit = {unit_used}, time_data length = {len(time_data)}")

    except Exception as e:
        messagebox.showerror("File Read Error", f"An error occurred while reading the file:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # Core XY coordinates
    xy = df[[xcol, ycol]].to_numpy()

    # Optional pos_y (height) if present and distinct from x/z
    pos_y = None
    pos_y_name = None
    for cand in ["y", "pos_y", "py", "height"]:
        if cand in df.columns and cand not in (xcol, ycol):
            try:
                pos_y = df[cand].astype(float).to_numpy()
                pos_y_name = cand
                break
            except Exception:
                pos_y = None
                pos_y_name = None

    # Optional speed: either explicit "speed" or from vx,vy,vz
    speeds = None
    cols_lower = {c.lower(): c for c in df.columns}
    if "speed" in cols_lower:
        try:
            speeds = df[cols_lower["speed"]].astype(float).to_numpy()
        except Exception:
            speeds = None
    else:
        vxc = cols_lower.get("vx")
        vyc = cols_lower.get("vy")
        vzc = cols_lower.get("vz")
        if vxc and vyc and vzc:
            try:
                vx = df[vxc].astype(float).to_numpy()
                vy = df[vyc].astype(float).to_numpy()
                vz = df[vzc].astype(float).to_numpy()
                speeds = np.sqrt(vx**2 + vy**2 + vz**2)
            except Exception:
                speeds = None

    # Optional gaze directions on ground plane
    gaze_dirs = None
    gx_col = cols_lower.get("gaze_dir_wx")
    gz_col = cols_lower.get("gaze_dir_wz")
    if gx_col and gz_col:
        try:
            gx = df[gx_col].astype(float).to_numpy()
            gz = df[gz_col].astype(float).to_numpy()
            gaze_dirs = np.stack([gx, gz], axis=1)
        except Exception:
            gaze_dirs = None

    # Optional video indices
    video_segment_idx = None
    video_frame_idx = None
    video_segments = None
    seg_col = cols_lower.get("video_segment")
    frm_col = cols_lower.get("video_frame")
    if seg_col and frm_col:
        try:
            video_segment_idx = df[seg_col].to_numpy()
            video_frame_idx = df[frm_col].to_numpy()
        except Exception:
            video_segment_idx = None
            video_frame_idx = None

        # Try to locate session folder and load video segments
        try:
            csv_path = Path(path)
            session_folder = csv_path.parent
            if session_folder.name.lower() == "processed":
                session_folder = session_folder.parent
            if session_folder.exists():
                video_segments = load_video_segments(session_folder)
                if not video_segments:
                    video_segments = None
        except Exception as e:
            print(f"Warning: could not load video segments: {e}")
            video_segments = None

    title = "EgoViz Dashboard"

    # Load annotation text
    annotation_text = ""
    if args.annotations and os.path.isfile(args.annotations):
        with open(args.annotations, 'r', encoding='utf-8') as f:
            annotation_text = f.read().strip()

    # Load per-frame annotations from annotations.json (h5 mode: row position == pose_idx).
    # each frame shows the most recent annotated second at or before it.
    frame_annotations = None
    if args.annotations_json and os.path.isfile(args.annotations_json) and args.session:
        import json, bisect
        try:
            allann = json.load(open(args.annotations_json, encoding='utf-8'))
            sess = sorted([r for r in allann if r.get('session') == args.session],
                          key=lambda r: r['pose_idx'])
            if sess:
                ann_idx = [r['pose_idx'] for r in sess]
                ann_txt = [r['annotation'] for r in sess]
                N = len(df)
                fa = [""] * N
                for pos in range(N):
                    j = bisect.bisect_right(ann_idx, pos) - 1
                    if j >= 0:
                        fa[pos] = ann_txt[j]
                frame_annotations = np.array(fa, dtype=object)
                print(f"INFO: loaded {len(sess)} annotations for {args.session}")
            else:
                print(f"WARNING: no annotations found for session {args.session}")
        except Exception as e:
            print(f"WARNING: could not load annotations.json: {e}")

    # session list + relaunch context for the dropdown (h5 mode only)
    session_list = None
    current_session = None
    relaunch_ctx = None
    if args.h5:
        try:
            import h5py
            session_list = sorted(list(h5py.File(args.h5, "r").keys()))
        except Exception:
            session_list = None
        current_session = args.session
        # derive a videos-root for relaunch: explicit flag, else parent of --videos
        vroot = args.videos_root
        if not vroot and args.videos:
            vroot = str(Path(args.videos).parent)
        relaunch_ctx = {
            "script": os.path.abspath(sys.argv[0]),
            "h5": os.path.abspath(args.h5),
            "videos_root": vroot,
            "annotations_json": os.path.abspath(args.annotations_json) if args.annotations_json else None,
        }

    app = TkinterCleaner(
        xy,
        headers=(xcol, ycol),
        time_data=time_data,
        title=title,
        neighbors=args.neighbors,
        pad_frac=args.pad,
        min_win=args.minwin,
        pos_y=pos_y,
        pos_y_name=pos_y_name,
        speeds=speeds,
        gaze_dirs=gaze_dirs,
        video_segments=video_segments,
        video_segment_idx=video_segment_idx,
        video_frame_idx=video_frame_idx,
        annotation_text=annotation_text,
        frame_annotations=frame_annotations,
        sessions=session_list,
        current_session=current_session,
        relaunch_ctx=relaunch_ctx,
    )
    if time_col:
        app.original_time_header = time_col
        
    # Keep original df & mask for saving
    app.original_df = df
    app.original_path = path
    app.keep_original_mask = np.ones(len(df), dtype=bool)

    app.mainloop()

if __name__ == "__main__":
    main()
