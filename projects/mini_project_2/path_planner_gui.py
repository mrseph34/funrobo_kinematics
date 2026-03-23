"""
Path Planner GUI — runs on your laptop
Start path_planner_server.py on the Pi first, then run this.
Set PI_HOST below to your Pi's IP or hostname.

Drawing plane axes:
  Grid horizontal (Y) = robot Y axis = arm swings left/right
  Grid vertical   (Z) = robot Z axis = arm moves up/down
  X offset param      = robot X axis = depth toward/away from you (default 0 = home depth)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import socket
import json
import threading
import math

PI_HOST = "pippi.local"
PI_PORT = 9999

GRID_SIZE_MM = 200
CANVAS_PX = 600
DEFAULT_X_OFFSET_MM = 0
DRAG_MIN_DIST = 8
MOVE_SPEED_MMS = 30

PRESETS = {
    "— select —": None,
    "Letter: A": [
        [(-30, -50), (0, 50), (30, -50)],
        [(-15, 0), (15, 0)],
    ],
    "Letter: N": [
        [(-30, -50), (-30, 50)],
        [(-30, 50), (30, -50)],
        [(30, -50), (30, 50)],
    ],
    "Letter: T": [
        [(-40, 50), (40, 50)],
        [(0, 50), (0, -50)],
    ],
    "Shape: Square": [
        [(60, 60), (-60, 60), (-60, -60), (60, -60), (60, 60)]
    ],
    "Shape: Star": [
        [(0, 60), (38, -60), (-60, 13), (60, 13), (-38, -60), (0, 60)]
    ],
    "Shape: Circle": [
        [(round(50*math.cos(a*math.pi/180)), round(50*math.sin(a*math.pi/180)))
         for a in range(0, 361, 15)]
    ],
}


def mm_to_px(y_mm, z_mm, canvas_size, grid_mm):
    scale = canvas_size / (2 * grid_mm)
    return canvas_size/2 + y_mm*scale, canvas_size/2 - z_mm*scale

def px_to_mm(px, py, canvas_size, grid_mm):
    scale = canvas_size / (2 * grid_mm)
    return (px - canvas_size/2) / scale, -(py - canvas_size/2) / scale


class PathPlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("5-DOF Arm — Cartesian Path Planner")
        self.root.configure(bg="#0f1117")
        self.root.resizable(False, False)

        self.waypoints = []
        self.drag_last = None
        self.executing = False
        self.sock = None
        self.buf = ""

        self.grid_mm = tk.IntVar(value=GRID_SIZE_MM)
        self.x_offset_mm = tk.DoubleVar(value=DEFAULT_X_OFFSET_MM)
        self.speed_mms = tk.DoubleVar(value=MOVE_SPEED_MMS)
        self.use_aik = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Not connected — click Connect")
        self.preset_var = tk.StringVar(value="— select —")
        self.pi_host = tk.StringVar(value=PI_HOST)

        self._build_ui()
        self._draw_grid()

    def _build_ui(self):
        BG = "#0f1117"
        PANEL = "#1a1d27"
        ACC = "#00e5ff"
        TXT = "#e0e0e0"
        DIM = "#555566"

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=PANEL)
        style.configure("Dark.TLabel", background=PANEL, foreground=TXT, font=("Courier", 10))
        style.configure("Title.TLabel", background=PANEL, foreground=ACC, font=("Courier", 11, "bold"))
        style.configure("Accent.TButton", background=ACC, foreground="#000000", font=("Courier", 10, "bold"), padding=6)
        style.map("Accent.TButton", background=[("active", "#00b8cc")])
        style.configure("Dim.TButton", background=PANEL, foreground=TXT, font=("Courier", 9), padding=4)
        style.map("Dim.TButton", background=[("active", "#2a2d3d")])

        outer = tk.Frame(self.root, bg=BG, padx=12, pady=12)
        outer.pack()

        tk.Label(outer, text="CARTESIAN PATH PLANNER", bg=BG, fg=ACC,
                 font=("Courier", 14, "bold")).pack(anchor="w", pady=(0, 8))

        content = tk.Frame(outer, bg=BG)
        content.pack()

        canvas_frame = tk.Frame(content, bg=PANEL, padx=2, pady=2)
        canvas_frame.pack(side="left", padx=(0, 12))

        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_PX, height=CANVAS_PX,
                                bg="#12141e", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.root.bind("<Control-z>", lambda e: self._undo())

        right = ttk.Frame(content, style="Dark.TFrame", padding=12)
        right.pack(side="left", fill="y")

        def section(label):
            f = ttk.Frame(right, style="Dark.TFrame")
            f.pack(fill="x", pady=(10, 4))
            ttk.Label(f, text=label, style="Title.TLabel").pack(anchor="w")
            tk.Frame(right, bg=ACC, height=1).pack(fill="x", pady=(0, 6))

        def param_row(label, var, minv, maxv, step=1.0):
            row = ttk.Frame(right, style="Dark.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, style="Dark.TLabel", width=18).pack(side="left")
            val_lbl = ttk.Label(row, style="Dark.TLabel", width=7)
            val_lbl.pack(side="right")
            def upd(*_): val_lbl.config(text=f"{var.get():.0f}")
            var.trace_add("write", upd)
            tk.Scale(row, variable=var, from_=minv, to=maxv, resolution=step,
                     orient="horizontal", length=140, bg=PANEL, fg=TXT,
                     troughcolor="#2a2d3d", activebackground=ACC,
                     highlightthickness=0, bd=0, showvalue=False,
                     command=lambda _: (upd(), self._draw_grid())).pack(side="left")
            upd()

        section("CONNECTION")
        conn_row = ttk.Frame(right, style="Dark.TFrame")
        conn_row.pack(fill="x", pady=2)
        ttk.Label(conn_row, text="Pi host:", style="Dark.TLabel").pack(side="left")
        tk.Entry(conn_row, textvariable=self.pi_host, width=16, bg="#2a2d3d", fg=TXT,
                 insertbackground=ACC, font=("Courier", 10), bd=0).pack(side="left", padx=6)

        btn_row = ttk.Frame(right, style="Dark.TFrame")
        btn_row.pack(fill="x", pady=4)
        ttk.Button(btn_row, text="Connect", style="Accent.TButton",
                   command=self._connect).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Disconnect", style="Dim.TButton",
                   command=self._disconnect).pack(side="left")

        self.conn_badge = tk.Label(right, text="● disconnected", bg=PANEL, fg="#ff4466",
                                   font=("Courier", 9, "bold"))
        self.conn_badge.pack(anchor="w", pady=2)

        section("PARAMETERS")
        param_row("Grid size (mm)", self.grid_mm, 50, 400, 10)
        param_row("X depth offset (mm)", self.x_offset_mm, -100, 100, 5)
        param_row("Speed (mm/s)", self.speed_mms, 5, 100, 5)

        info = ttk.Frame(right, style="Dark.TFrame")
        info.pack(fill="x", pady=2)
        ttk.Label(info, text="Grid Y = left/right", style="Dark.TLabel").pack(anchor="w")
        ttk.Label(info, text="Grid Z = up/down", style="Dark.TLabel").pack(anchor="w")
        ttk.Label(info, text="X offset = depth from home", style="Dark.TLabel").pack(anchor="w")

        ik_row = ttk.Frame(right, style="Dark.TFrame")
        ik_row.pack(fill="x", pady=6)
        ttk.Label(ik_row, text="IK solver:", style="Dark.TLabel", width=18).pack(side="left")
        for txt, val in [("AIK", True), ("NIK", False)]:
            tk.Radiobutton(ik_row, text=txt, variable=self.use_aik, value=val,
                           bg=PANEL, fg=TXT, selectcolor="#2a2d3d",
                           activebackground=PANEL, font=("Courier", 9)).pack(side="left")

        section("PRESETS")
        preset_row = ttk.Frame(right, style="Dark.TFrame")
        preset_row.pack(fill="x", pady=2)
        ttk.Label(preset_row, text="Load preset:", style="Dark.TLabel").pack(side="left")
        pm = ttk.Combobox(preset_row, textvariable=self.preset_var,
                          values=list(PRESETS.keys()), state="readonly", width=18)
        pm.pack(side="left", padx=6)
        pm.bind("<<ComboboxSelected>>", self._on_preset_select)

        section("MANUAL WAYPOINT")
        manual = ttk.Frame(right, style="Dark.TFrame")
        manual.pack(fill="x", pady=2)
        ttk.Label(manual, text="Y (mm):", style="Dark.TLabel").pack(side="left")
        self.manual_y = tk.Entry(manual, width=6, bg="#2a2d3d", fg=TXT,
                                 insertbackground=ACC, font=("Courier", 10), bd=0)
        self.manual_y.pack(side="left", padx=4)
        ttk.Label(manual, text="Z (mm):", style="Dark.TLabel").pack(side="left")
        self.manual_z = tk.Entry(manual, width=6, bg="#2a2d3d", fg=TXT,
                                 insertbackground=ACC, font=("Courier", 10), bd=0)
        self.manual_z.pack(side="left", padx=4)
        ttk.Button(manual, text="Add", style="Dim.TButton",
                   command=self._add_manual).pack(side="left", padx=4)

        self.wp_listbox = tk.Listbox(right, height=8, bg="#12141e", fg=TXT,
                                     selectbackground=ACC, selectforeground="#000",
                                     font=("Courier", 9), bd=0,
                                     highlightthickness=1, highlightcolor=DIM,
                                     activestyle="none")
        self.wp_listbox.pack(fill="x", pady=6)
        ttk.Button(right, text="Remove selected", style="Dim.TButton",
                   command=self._remove_selected).pack(fill="x", pady=2)

        section("ACTIONS")
        ttk.Button(right, text="▶  EXECUTE PATH", style="Accent.TButton", command=self._execute).pack(fill="x", pady=4)
        ttk.Button(right, text="⏹  STOP", style="Dim.TButton", command=self._stop).pack(fill="x", pady=2)
        ttk.Button(right, text="⌂  Home", style="Dim.TButton", command=self._home).pack(fill="x", pady=2)
        ttk.Button(right, text="✕  Clear all", style="Dim.TButton", command=self._clear).pack(fill="x", pady=2)
        ttk.Button(right, text="↩  Undo", style="Dim.TButton", command=self._undo).pack(fill="x", pady=2)

        tk.Label(outer, textvariable=self.status, bg=BG, fg="#888899",
                 font=("Courier", 9), anchor="w").pack(fill="x", pady=(8, 0))

    def _draw_grid(self, *_):
        self.canvas.delete("all")
        grid = self.grid_mm.get()
        ACC = "#00e5ff"

        for i in range(11):
            frac = i / 10
            x = frac * CANVAS_PX
            col = "#252840" if i % 2 == 0 else "#1e2130"
            self.canvas.create_line(x, 0, x, CANVAS_PX, fill=col, width=1)
            self.canvas.create_line(0, x, CANVAS_PX, x, fill=col, width=1)

        cx = cy = CANVAS_PX / 2
        self.canvas.create_line(0, cy, CANVAS_PX, cy, fill="#334455", width=1, dash=(4, 4))
        self.canvas.create_line(cx, 0, cx, CANVAS_PX, fill="#334455", width=1, dash=(4, 4))
        self.canvas.create_text(CANVAS_PX-8, cy+12, text=f"+Y left/right ({grid}mm)", fill=ACC, font=("Courier", 8), anchor="e")
        self.canvas.create_text(cx+8, 8, text=f"+Z up/down ({grid}mm)", fill=ACC, font=("Courier", 8), anchor="w")

        for i in range(-5, 6):
            if i == 0:
                continue
            val = int(i * grid / 5)
            px, _ = mm_to_px(val, 0, CANVAS_PX, grid)
            _, pz = mm_to_px(0, val, CANVAS_PX, grid)
            self.canvas.create_text(px, cy-8, text=str(val), fill="#445566", font=("Courier", 7))
            self.canvas.create_text(cx+4, pz, text=str(val), fill="#445566", font=("Courier", 7), anchor="w")

        self._redraw_waypoints()

    def _redraw_waypoints(self):
        self.canvas.delete("waypoints")
        self.canvas.delete("path")
        grid = self.grid_mm.get()
        COLORS = ["#00e5ff", "#ff6b35", "#a8ff3e", "#ff3ea8"]
        stroke_idx = 0
        stroke_pts = []
        first_real = True

        def flush(pts, color):
            if len(pts) >= 2:
                for i in range(len(pts)-1):
                    self.canvas.create_line(*pts[i], *pts[i+1], fill=color, width=2, tags="path")

        for wp in self.waypoints:
            if wp is None:
                flush(stroke_pts, COLORS[stroke_idx % len(COLORS)])
                stroke_idx += 1
                stroke_pts = []
                continue
            y, z = wp
            px, pz = mm_to_px(y, z, CANVAS_PX, grid)
            stroke_pts.append((px, pz))
            color = "#00ff88" if first_real else "#ff4466"
            r = 7 if first_real else 4
            self.canvas.create_oval(px-r, pz-r, px+r, pz+r, fill=color, outline="", tags="waypoints")
            first_real = False

        flush(stroke_pts, COLORS[stroke_idx % len(COLORS)])

    def _refresh_listbox(self):
        self.wp_listbox.delete(0, "end")
        count = 1
        for wp in self.waypoints:
            if wp is None:
                self.wp_listbox.insert("end", "  ── pen lift ──")
            else:
                self.wp_listbox.insert("end", f"  {count:2d}:  Y={wp[0]:+7.1f}  Z={wp[1]:+7.1f}")
                count += 1

    def _on_click(self, event):
        if self.executing:
            return
        y, z = px_to_mm(event.x, event.y, CANVAS_PX, self.grid_mm.get())
        self._add_waypoint(y, z)
        self.drag_last = (event.x, event.y)

    def _on_drag(self, event):
        if self.executing or self.drag_last is None:
            return
        dx = event.x - self.drag_last[0]
        dz = event.y - self.drag_last[1]
        if math.sqrt(dx*dx + dz*dz) >= DRAG_MIN_DIST:
            y, z = px_to_mm(event.x, event.y, CANVAS_PX, self.grid_mm.get())
            self._add_waypoint(y, z)
            self.drag_last = (event.x, event.y)

    def _on_release(self, event):
        self.drag_last = None

    def _on_right_click(self, event):
        real = [(i, w) for i, w in enumerate(self.waypoints) if w is not None]
        if not real:
            return
        my, mz = px_to_mm(event.x, event.y, CANVAS_PX, self.grid_mm.get())
        idx = min(real, key=lambda iw: math.sqrt((iw[1][0]-my)**2 + (iw[1][1]-mz)**2))[0]
        self.waypoints.pop(idx)
        self._redraw_waypoints()
        self._refresh_listbox()

    def _add_waypoint(self, y_mm, z_mm):
        self.waypoints.append((round(y_mm, 1), round(z_mm, 1)))
        self._redraw_waypoints()
        self._refresh_listbox()
        self.status.set(f"Waypoint {sum(1 for w in self.waypoints if w is not None)}: Y={y_mm:.1f}mm  Z={z_mm:.1f}mm")

    def _add_manual(self):
        try:
            y = float(self.manual_y.get())
            z = float(self.manual_z.get())
            self._add_waypoint(y, z)
            self.manual_y.delete(0, "end")
            self.manual_z.delete(0, "end")
        except ValueError:
            messagebox.showerror("Invalid input", "Y and Z must be numbers.")

    def _remove_selected(self):
        sel = self.wp_listbox.curselection()
        if sel:
            self.waypoints.pop(sel[0])
            self._redraw_waypoints()
            self._refresh_listbox()

    def _undo(self):
        if self.waypoints:
            self.waypoints.pop()
            self._redraw_waypoints()
            self._refresh_listbox()

    def _clear(self):
        self.waypoints.clear()
        self._draw_grid()
        self._refresh_listbox()
        self.status.set("Cleared.")

    def _on_preset_select(self, _event):
        name = self.preset_var.get()
        strokes = PRESETS.get(name)
        if not strokes:
            return
        self.waypoints.clear()
        for i, stroke in enumerate(strokes):
            for pt in stroke:
                self.waypoints.append((float(pt[0]), float(pt[1])))
            if i < len(strokes) - 1:
                self.waypoints.append(None)
        self._redraw_waypoints()
        self._refresh_listbox()
        n = sum(1 for w in self.waypoints if w is not None)
        self.status.set(f"Loaded '{name}' — {n} waypoints, {len(strokes)} stroke(s)")
        self.preset_var.set("— select —")

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.pi_host.get(), PI_PORT))
            self.sock.settimeout(0.1)
            self.conn_badge.config(text="● connected", fg="#00ff88")
            self.status.set(f"Connected to {self.pi_host.get()}")
            threading.Thread(target=self._listen, daemon=True).start()
        except Exception as e:
            self.status.set(f"Connection failed: {e}")

    def _disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        self.conn_badge.config(text="● disconnected", fg="#ff4466")
        self.status.set("Disconnected.")

    def _listen(self):
        while self.sock:
            try:
                data = self.sock.recv(4096).decode()
                if data:
                    self.buf += data
                    while "\n" in self.buf:
                        line, self.buf = self.buf.split("\n", 1)
                        msg = json.loads(line)
                        if msg.get("status") == "done":
                            self.executing = False
                            self.status.set("Path complete!")
                        elif msg.get("status") == "at":
                            wp = msg["wp"]
                            self.status.set(f"At Y={wp[0]:.0f}mm  Z={wp[1]:.0f}mm")
                        elif msg.get("status") == "stopped":
                            self.executing = False
                            self.status.set("Stopped.")
                        elif msg.get("status") == "homed":
                            self.status.set("Homed.")
            except socket.timeout:
                continue
            except Exception:
                break

    def _send(self, msg):
        if not self.sock:
            messagebox.showerror("Not connected", "Connect to the Pi first.")
            return False
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        return True

    def _execute(self):
        if len([w for w in self.waypoints if w is not None]) < 2:
            messagebox.showwarning("Not enough waypoints", "Add at least 2 waypoints.")
            return
        if self.executing:
            return
        wp_serialized = [list(w) if w is not None else None for w in self.waypoints]
        if self._send({
            "cmd": "run",
            "waypoints": wp_serialized,
            "x_offset_mm": self.x_offset_mm.get(),
            "speed_mms": self.speed_mms.get(),
            "use_aik": self.use_aik.get(),
        }):
            self.executing = True
            self.status.set("Executing...")

    def _stop(self):
        self._send({"cmd": "stop"})

    def _home(self):
        self._send({"cmd": "home"})


if __name__ == "__main__":
    root = tk.Tk()
    PathPlannerGUI(root)
    root.mainloop()