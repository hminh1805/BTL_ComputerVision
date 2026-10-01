"""
Computer Vision App – GUI với Tkinter | Light Theme
"""

import os
import sys

# Fix Tcl/Tk for Conda/Miniconda on Windows
if "TCL_LIBRARY" not in os.environ:
    for p in [os.path.join(sys.base_prefix, "Library", "lib", "tcl8.6"),
              os.path.join(sys.base_prefix, "tcl", "tcl8.6")]:
        if os.path.exists(p):
            os.environ["TCL_LIBRARY"] = p
            break

if "TK_LIBRARY" not in os.environ:
    for p in [os.path.join(sys.base_prefix, "Library", "lib", "tk8.6"),
              os.path.join(sys.base_prefix, "tcl", "tk8.6")]:
        if os.path.exists(p):
            os.environ["TK_LIBRARY"] = p
            break

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk

from src.ImageLoader import ImageLoader
from src.Filter import Filter
from src.Transform import Transform
from src.utils import load_image

# ══════════════════════════════════════════════════════════════════════════════
# LIGHT THEME TOKENS
# ══════════════════════════════════════════════════════════════════════════════
BG           = "#F5F6FA"
PANEL        = "#FFFFFF"
CARD         = "#F0F2F8"
SIDEBAR      = "#EEF0F7"

ACCENT       = "#5C6BC0"
ACCENT_DARK  = "#3949AB"
ACCENT_LIGHT = "#E8EAF6"
SUCCESS      = "#2E7D32"
SUCCESS_BG   = "#E8F5E9"
WARNING      = "#E65100"
DANGER       = "#C62828"

TEXT         = "#212121"
TEXT_MED     = "#424242"
TEXT_DIM     = "#757575"
BORDER       = "#C5CAE9"
HOVER        = "#E8EAF6"

FONT_TITLE   = ("Segoe UI", 14, "bold")
FONT_HEAD    = ("Segoe UI", 10, "bold")
FONT_NORMAL  = ("Segoe UI", 9)
FONT_SMALL   = ("Segoe UI", 8)
FONT_BTN     = ("Segoe UI", 9, "bold")

PREVIEW_W, PREVIEW_H = 380, 300
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


# ══════════════════════════════════════════════════════════════════════════════
# HELPER WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
def make_btn(parent, text, command, style="primary", width=None):
    palettes = {
        "primary": (ACCENT,      ACCENT_DARK, "white",   "white"),
        "success": (SUCCESS,     "#1B5E20",   "white",   "white"),
        "outline": (PANEL,       HOVER,       ACCENT,    ACCENT),
        "subtle":  (CARD,        BORDER,      TEXT_MED,  TEXT_MED),
        "danger":  (DANGER,      "#B71C1C",   "white",   "white"),
    }
    bg, hover_bg, fg, hover_fg = palettes.get(style, palettes["primary"])
    kw = dict(bg=bg, fg=fg, font=FONT_BTN, relief="flat",
              cursor="hand2", bd=0, padx=10, pady=5,
              activebackground=hover_bg, activeforeground=hover_fg,
              command=command)
    if width:
        kw["width"] = width
    b = tk.Button(parent, text=text, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=hover_bg))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b


def make_label(parent, text, font=FONT_NORMAL, fg=TEXT, bg=None, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg or PANEL, **kw)


def section_header(parent, text, bg=SIDEBAR):
    f = tk.Frame(parent, bg=bg)
    tk.Frame(f, bg=ACCENT, width=3).pack(side="left", fill="y", padx=(0, 8))
    tk.Label(f, text=text, font=FONT_HEAD, fg=ACCENT, bg=bg).pack(side="left", anchor="w")
    return f


def hsep(parent, bg=SIDEBAR, pady=5):
    f = tk.Frame(parent, bg=BORDER, height=1)
    f.pack(fill="x", padx=10, pady=pady)
    return f


def img_to_photoimage(arr, max_w=PREVIEW_W, max_h=PREVIEW_H):
    if arr is None:
        return None
    if arr.ndim == 2:
        pil = PILImage.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 1:
        pil = PILImage.fromarray(arr[:, :, 0].astype(np.uint8), mode="L").convert("RGB")
    else:
        pil = PILImage.fromarray(arr.astype(np.uint8), mode="RGB")
    pil.thumbnail((max_w, max_h), PILImage.LANCZOS)
    return ImageTk.PhotoImage(pil)


def expand_paths(paths):
    result = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if os.path.isfile(fp) and fp.lower().endswith(IMG_EXTS):
                    result.append(fp)
        elif os.path.isfile(p) and p.lower().endswith(IMG_EXTS):
            result.append(p)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STATUS BAR
# ══════════════════════════════════════════════════════════════════════════════
class StatusBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD, bd=0,
                         highlightthickness=1, highlightbackground=BORDER, **kw)
        self.lbl = tk.Label(self, text="  Sẵn sàng.", font=FONT_SMALL,
                             fg=TEXT_DIM, bg=CARD, anchor="w")
        self.lbl.pack(fill="x", padx=8, pady=3)

    def set(self, msg, color=TEXT_DIM):
        self.lbl.config(text="  " + msg, fg=color)
        self.update_idletasks()

    def ok(self, msg):   self.set("✔  " + msg, SUCCESS)
    def err(self, msg):  self.set("✖  " + msg, DANGER)
    def info(self, msg): self.set("⟳  " + msg, WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# IO SECTION (Input + Output settings)
# ══════════════════════════════════════════════════════════════════════════════
class IOSection(tk.Frame):
    def __init__(self, parent, on_input_change, bg_color=SIDEBAR, **kw):
        super().__init__(parent, bg=bg_color, **kw)
        self.on_input_change = on_input_change
        self.input_paths = []
        self._bg = bg_color
        self.save_mode = tk.StringVar(value="combined")

        # Input row
        row1 = tk.Frame(self, bg=bg_color)
        row1.pack(fill="x", padx=12, pady=(8, 3))
        tk.Label(row1, text="Đầu vào:", font=FONT_HEAD,
                 fg=TEXT_MED, bg=bg_color, width=9, anchor="w").pack(side="left")
        self.inp_var = tk.StringVar(value="Chưa chọn...")
        tk.Entry(row1, textvariable=self.inp_var, font=FONT_SMALL,
                 bg=CARD, fg=TEXT_DIM, relief="solid", bd=1,
                 state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 6))
        make_btn(row1, "📄 Ảnh",   self._pick_file, style="outline").pack(side="left", padx=(0, 4))
        make_btn(row1, "📁 Folder", self._pick_dir,  style="outline").pack(side="left")

        # Output row
        row2 = tk.Frame(self, bg=bg_color)
        row2.pack(fill="x", padx=12, pady=3)
        tk.Label(row2, text="Lưu vào:", font=FONT_HEAD,
                 fg=TEXT_MED, bg=bg_color, width=9, anchor="w").pack(side="left")
        self.out_var = tk.StringVar(value="test_sample/")
        tk.Entry(row2, textvariable=self.out_var, font=FONT_SMALL,
                 bg=CARD, fg=TEXT_MED, relief="solid", bd=1,
                 ).pack(side="left", expand=True, fill="x", padx=(0, 6))
        make_btn(row2, "📁 Chọn", self._pick_out_dir, style="outline").pack(side="left")

        # Save mode – two toggle buttons
        row3 = tk.Frame(self, bg=bg_color)
        row3.pack(fill="x", padx=12, pady=(4, 8))
        tk.Label(row3, text="Chế độ lưu:", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=bg_color).pack(anchor="w", pady=(0, 4))
        btn_row = tk.Frame(row3, bg=bg_color)
        btn_row.pack(fill="x")
        self._mode_btns = {}
        for txt, val in [("🖼 Gốc + Kết quả", "combined"), ("💾 Chỉ kết quả", "output_only")]:
            b = tk.Button(btn_row, text=txt, font=FONT_SMALL, relief="solid", bd=1,
                          padx=8, pady=4, cursor="hand2",
                          command=lambda v=val: self._set_mode(v))
            b.pack(side="left", padx=(0, 6))
            self._mode_btns[val] = b
        self._update_mode_btns()

    def _set_mode(self, val):
        self.save_mode.set(val)
        self._update_mode_btns()

    def _update_mode_btns(self):
        cur = self.save_mode.get()
        for val, btn in self._mode_btns.items():
            if val == cur:
                btn.config(bg=ACCENT, fg="white", activebackground=ACCENT_DARK,
                           activeforeground="white")
            else:
                btn.config(bg=CARD, fg=TEXT_MED, activebackground=BORDER,
                           activeforeground=TEXT)

    def _pick_file(self):
        paths = filedialog.askopenfilenames(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")])
        if paths:
            self.input_paths = list(paths)
            self.inp_var.set("; ".join(os.path.basename(p) for p in paths))
            self.on_input_change(self.input_paths)

    def _pick_dir(self):
        d = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if d:
            self.input_paths = [d]
            self.inp_var.set(d)
            self.on_input_change(self.input_paths)

    def set_paths(self, paths):
        """Đồng bộ hiển thị đường dẫn khi tab khác broadcast, không trigger on_input_change."""
        self.input_paths = list(paths)
        if not paths:
            self.inp_var.set("Chưa chọn...")
        elif len(paths) == 1 and os.path.isdir(paths[0]):
            self.inp_var.set(paths[0])
        else:
            self.inp_var.set("; ".join(os.path.basename(p) for p in paths))

    def _pick_out_dir(self):
        d = filedialog.askdirectory(title="Chọn thư mục lưu")
        if d:
            self.out_var.set(d)

    def get_output_dir(self):
        return self.out_var.get() or "test_sample"

    def get_save_mode(self):
        return self.save_mode.get()



# ══════════════════════════════════════════════════════════════════════════════
# PREVIEW PANEL
# ══════════════════════════════════════════════════════════════════════════════
class PreviewPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._orig_tk = None
        self._res_tk  = None

        for side, attr, lbl_text, lbl_color in [
            ("left",  "orig_lbl", "Ảnh gốc (Original)",  TEXT_DIM),
            ("right", "res_lbl",  "Kết quả (Result)",     SUCCESS),
        ]:
            card = tk.Frame(self, bg=PANEL, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
            card.pack(side="left", expand=True, fill="both",
                      padx=(0, 4) if side == "left" else (4, 0))
            tk.Label(card, text=lbl_text, font=FONT_SMALL,
                     fg=lbl_color, bg=PANEL).pack(pady=(8, 2))
            default_txt = "Chưa có ảnh" if side == "left" else "Chưa có kết quả"
            lbl = tk.Label(card, bg=CARD, text=default_txt,
                           fg=TEXT_DIM, font=FONT_SMALL)
            lbl.pack(expand=True, fill="both", padx=6, pady=(0, 8))
            setattr(self, attr, lbl)

    def set_images(self, orig_arr, result_arr):
        self._orig_tk = img_to_photoimage(orig_arr)
        self._res_tk  = img_to_photoimage(result_arr)
        if self._orig_tk:
            self.orig_lbl.config(image=self._orig_tk, text="")
        else:
            self.orig_lbl.config(image="", text="Chưa có ảnh")
        if self._res_tk:
            self.res_lbl.config(image=self._res_tk, text="")
        else:
            self.res_lbl.config(image="", text="Chưa có kết quả")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LIST BOX with scrollbar
# ══════════════════════════════════════════════════════════════════════════════
class ImageListBox(tk.Frame):
    def __init__(self, parent, on_select, height=6, bg_color=SIDEBAR, **kw):
        super().__init__(parent, bg=bg_color, **kw)
        self.on_select = on_select
        sb = tk.Scrollbar(self, orient="vertical")
        self.listbox = tk.Listbox(
            self, yscrollcommand=sb.set,
            bg=CARD, fg=TEXT, selectbackground=ACCENT_LIGHT,
            selectforeground=ACCENT, font=FONT_SMALL,
            relief="solid", bd=1, height=height,
            activestyle="none", highlightthickness=0)
        sb.config(command=self.listbox.yview)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_sel)

    def _on_sel(self, _e):
        sel = self.listbox.curselection()
        if sel:
            self.on_select(sel[0])

    def set_items(self, names):
        self.listbox.delete(0, "end")
        for n in names:
            self.listbox.insert("end", n)

    def select(self, idx):
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(idx)
        self.listbox.see(idx)


# ══════════════════════════════════════════════════════════════════════════════
# OPTION BUTTON GROUP (Button selectors with active state)
# ══════════════════════════════════════════════════════════════════════════════
class OptionButtonGroup(tk.Frame):
    def __init__(self, parent, options, variable, on_change=None, bg_color=SIDEBAR, **kw):
        super().__init__(parent, bg=bg_color, **kw)
        self.variable = variable
        self.on_change = on_change
        self.buttons = {}

        for text, val in options:
            btn = tk.Button(
                self, text=f"  {text}", font=FONT_NORMAL, anchor="w", padx=10, pady=5,
                relief="solid", bd=1, highlightthickness=0, cursor="hand2"
            )
            btn.config(command=lambda v=val: self._select(v))
            btn.pack(fill="x", pady=2, padx=2)
            self.buttons[val] = btn

        self.variable.trace_add("write", lambda *_: self.update_states())
        self.update_states()

    def _select(self, val):
        self.variable.set(val)
        self.update_states()
        if self.on_change:
            self.on_change()

    def update_states(self):
        cur = self.variable.get()
        for val, btn in self.buttons.items():
            txt = btn.cget("text").lstrip("✓ ").strip()
            if val == cur:
                btn.config(
                    text=f"✓  {txt}",
                    bg=ACCENT, fg="white", activebackground=ACCENT_DARK,
                    activeforeground="white", font=FONT_BTN,
                    highlightbackground=ACCENT
                )
            else:
                btn.config(
                    text=f"    {txt}",
                    bg=CARD, fg=TEXT_MED, activebackground=BORDER,
                    activeforeground=TEXT, font=FONT_NORMAL,
                    highlightbackground=BORDER
                )


# ══════════════════════════════════════════════════════════════════════════════
# SCROLLABLE SIDEBAR CONTAINER
# ══════════════════════════════════════════════════════════════════════════════
class ScrollableSidebar(tk.Frame):
    """Thanh bên (Sidebar) tự động hiển thị thanh cuộn dọc (Scrollbar) khi nội dung quá dài."""
    def __init__(self, parent, width=290, bg=SIDEBAR, **kw):
        super().__init__(parent, bg=bg, width=width, **kw)
        self.pack_propagate(False)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        self.content = tk.Frame(self.canvas, bg=bg)

        self.content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.bind_mousewheel(self.content)

    def bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        for child in widget.winfo_children():
            self.bind_mousewheel(child)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def save_result(orig_arr, result_arr, out_dir, save_mode, filename):
    os.makedirs(out_dir, exist_ok=True)
    stem, _ = os.path.splitext(os.path.basename(filename))

    def to_bgr(arr):
        a = arr.astype(np.uint8)
        if a.ndim == 2:
            return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        if a.ndim == 3 and a.shape[2] == 1:
            return cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)

    if save_mode == "combined":
        ob, rb = to_bgr(orig_arr), to_bgr(result_arr)
        h = max(ob.shape[0], rb.shape[0])
        def pad_h(img):
            if img.shape[0] == h:
                return img
            return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        combined = np.hstack([pad_h(ob), pad_h(rb)])
        path = os.path.join(out_dir, f"{stem}_comparison.jpg")
        cv2.imwrite(path, combined)
    else:
        path = os.path.join(out_dir, f"{stem}_result.jpg")
        cv2.imwrite(path, to_bgr(result_arr))
    return path


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – COLOR REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════
class ColorTab(tk.Frame):
    def __init__(self, parent, status, app=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.status = status
        self.app = app
        self.loader = ImageLoader()
        self._imgs = []
        self._cur_orig = None
        self._cur_result = None
        self._cur_path = ""

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = ScrollableSidebar(self, width=290, bg=SIDEBAR)
        sidebar.pack(side="left", fill="y")
        sb = sidebar.content

        tk.Label(sb, text="🎨  Màu & Kênh màu",
                 font=FONT_TITLE, fg=ACCENT, bg=SIDEBAR).pack(pady=(14, 2), padx=14, anchor="w")
        tk.Label(sb, text="Chuyển đổi không gian màu, tách kênh",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=SIDEBAR).pack(padx=14, anchor="w")
        hsep(sb)

        self.io = IOSection(sb, self._on_input, bg_color=SIDEBAR)
        self.io.pack(fill="x")
        hsep(sb)

        section_header(sb, "Phép biến đổi").pack(fill="x", padx=12, pady=(4, 6))
        self.op_var = tk.StringVar(value="grayscale")
        opts = [("RGB → Grayscale", "grayscale"),
                ("Tách kênh R",     "channel_r"),
                ("Tách kênh G",     "channel_g"),
                ("Tách kênh B",     "channel_b"),
                ("Hoán đổi BGR",    "swap_bgr")]
        OptionButtonGroup(sb, opts, self.op_var, on_change=self._run, bg_color=SIDEBAR).pack(fill="x", padx=12, pady=(0, 4))
        hsep(sb)

        section_header(sb, "Danh sách ảnh").pack(fill="x", padx=12, pady=(4, 4))
        self.img_list = ImageListBox(sb, self._on_select, height=5)
        self.img_list.pack(fill="x", padx=12, pady=(0, 6))

        btn_r = tk.Frame(sb, bg=SIDEBAR)
        btn_r.pack(fill="x", padx=12, pady=(0, 10))

        row1 = tk.Frame(btn_r, bg=SIDEBAR)
        row1.pack(fill="x", pady=(0, 4))
        make_btn(row1, "▶  Thực hiện", self._run, style="primary").pack(side="left", expand=True, fill="x", padx=(0, 4))
        make_btn(row1, "💾  Lưu ảnh này", self._save, style="success").pack(side="left", expand=True, fill="x")

        make_btn(btn_r, "⚡  Biến đổi & Lưu tất cả", self._save_all, style="outline").pack(fill="x")

        self.preview = PreviewPanel(self)
        self.preview.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    def _on_input(self, paths):
        self.load_paths(paths)
        if self.app:
            self.app.sync_input_paths(paths, sender_tab=self)

    def load_paths(self, paths):
        """Load ảnh từ paths (raw: file list or folder list). Cập nhật listbox và preview."""
        self.io.set_paths(paths)
        all_p = expand_paths(paths)
        self._imgs = []
        for p in all_p:
            try:
                self._imgs.extend(self.loader.load_image(p))
            except Exception as e:
                self.status.err(f"Load lỗi: {e}")
        self.img_list.set_items([os.path.basename(i.path) for i in self._imgs])
        if self._imgs:
            self.img_list.select(0)
            self._cur_path = self._imgs[0].path
            self._run()

    def _on_select(self, idx):
        if self._imgs and idx < len(self._imgs):
            self._cur_path = self._imgs[idx].path
            self._run()

    def _run(self):
        if not self._imgs:
            return
        idx = next((i for i, x in enumerate(self._imgs) if x.path == self._cur_path), 0)
        img_obj = self._imgs[idx]
        self._cur_orig = img_obj.data
        op = self.op_var.get()
        try:
            if op == "grayscale":
                out = self.loader.rgb_to_gray(img_obj)
                result = out.data if out else None
            elif op == "channel_r":
                r, g, b = self.loader.split_channels(img_obj)
                result = r.data if r else None
            elif op == "channel_g":
                r, g, b = self.loader.split_channels(img_obj)
                result = g.data if g else None
            elif op == "channel_b":
                r, g, b = self.loader.split_channels(img_obj)
                result = b.data if b else None
            elif op == "swap_bgr":
                r, g, b = self.loader.split_channels(img_obj)
                sw = self.loader.merge_channels(b, g, r, output_name="_tmp_swap.jpg")
                result = sw.data if sw else None
            else:
                result = None
            self._cur_result = result
            self.preview.set_images(self._cur_orig, result)
            self.status.ok(f"[{op}] — {os.path.basename(img_obj.path)}")
        except Exception as e:
            self.status.err(str(e))

    def _save(self):
        if self._cur_result is None:
            messagebox.showwarning("Chưa có kết quả", "Chạy phép biến đổi trước!")
            return
        try:
            path = save_result(self._cur_orig, self._cur_result,
                               self.io.get_output_dir(), self.io.get_save_mode(),
                               self._cur_path or "color_output.jpg")
            self.status.ok(f"Đã lưu → {path}")
            messagebox.showinfo("Đã lưu", f"Kết quả:\n{path}")
        except Exception as e:
            self.status.err(str(e))

    def _save_all(self):
        if not self._imgs:
            messagebox.showwarning("Chưa có danh sách", "Chọn thư mục ảnh trước!")
            return
        op = self.op_var.get()
        out_dir = self.io.get_output_dir()
        save_mode = self.io.get_save_mode()

        def _task():
            count = 0
            try:
                for img_obj in self._imgs:
                    orig = img_obj.data
                    if op == "grayscale":
                        out = self.loader.rgb_to_gray(img_obj)
                        res = out.data if out else None
                    elif op == "channel_r":
                        r, _, _ = self.loader.split_channels(img_obj)
                        res = r.data if r else None
                    elif op == "channel_g":
                        _, g, _ = self.loader.split_channels(img_obj)
                        res = g.data if g else None
                    elif op == "channel_b":
                        _, _, b = self.loader.split_channels(img_obj)
                        res = b.data if b else None
                    elif op == "swap_bgr":
                        r, g, b = self.loader.split_channels(img_obj)
                        sw = self.loader.merge_channels(b, g, r, output_name="_tmp.jpg")
                        res = sw.data if sw else None
                    else:
                        res = None
                    if res is not None:
                        save_result(orig, res, out_dir, save_mode, img_obj.path)
                        count += 1
                self.after(0, lambda: self.status.ok(f"Đã lưu {count} ảnh → {out_dir}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Hoàn thành", f"Đã xử lý & lưu {count} ảnh vào:\n{out_dir}"))
            except Exception as e:
                self.after(0, lambda err=e: self.status.err(str(err)))
        threading.Thread(target=_task, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – FILTER
# ══════════════════════════════════════════════════════════════════════════════
class FilterTab(tk.Frame):
    def __init__(self, parent, status, app=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.status = status
        self.app = app
        self.filter_tool = Filter()
        self._cur_orig = None
        self._cur_result = None
        self._cur_path = ""
        self._img_paths = []

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = ScrollableSidebar(self, width=295, bg=SIDEBAR)
        sidebar.pack(side="left", fill="y")
        sb = sidebar.content

        tk.Label(sb, text="🔍  Lọc ảnh (Filter)",
                 font=FONT_TITLE, fg=ACCENT, bg=SIDEBAR).pack(pady=(14, 2), padx=14, anchor="w")
        tk.Label(sb, text="Low-pass và High-pass filters",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=SIDEBAR).pack(padx=14, anchor="w")
        hsep(sb)

        self.io = IOSection(sb, self._on_input, bg_color=SIDEBAR)
        self.io.pack(fill="x")
        hsep(sb)

        # Filter options
        section_header(sb, "Bộ lọc").pack(fill="x", padx=12, pady=(4, 6))
        self.filter_var = tk.StringVar(value="mean")
        self.filter_var.trace_add("write", lambda *_: self._on_filter_change())
        filter_opts = [
            ("Mean Filter",        "mean"),
            ("Gaussian Filter",    "gaussian"),
            ("Sobel X",            "sobel_x"),
            ("Sobel Y",            "sobel_y"),
            ("Sobel Magnitude",    "sobel_both"),
            ("Laplacian (4-conn)", "laplacian_4"),
            ("Laplacian (8-conn)", "laplacian_8")
        ]
        OptionButtonGroup(sb, filter_opts, self.filter_var, bg_color=SIDEBAR).pack(fill="x", padx=12, pady=(0, 4))
        hsep(sb)

        # Dynamic params section
        section_header(sb, "Tham số").pack(fill="x", padx=12, pady=(4, 4))
        self.param_frame = tk.Frame(sb, bg=SIDEBAR)
        self.param_frame.pack(fill="x", padx=14, pady=(0, 6))
        self.kernel_var = tk.IntVar(value=3)
        self.sigma_var  = tk.DoubleVar(value=1.0)
        self._build_param_widgets()
        hsep(sb)

        # Image list
        section_header(sb, "Danh sách ảnh").pack(fill="x", padx=12, pady=(4, 4))
        self.img_list = ImageListBox(sb, self._on_select, height=4)
        self.img_list.pack(fill="x", padx=12, pady=(0, 6))

        btn_r = tk.Frame(sb, bg=SIDEBAR)
        btn_r.pack(fill="x", padx=12, pady=(0, 10))

        row1 = tk.Frame(btn_r, bg=SIDEBAR)
        row1.pack(fill="x", pady=(0, 4))
        make_btn(row1, "▶  Lọc ảnh", self._run, style="primary").pack(side="left", expand=True, fill="x", padx=(0, 4))
        make_btn(row1, "💾  Lưu ảnh này", self._save, style="success").pack(side="left", expand=True, fill="x")

        make_btn(btn_r, "⚡  Lọc & Lưu tất cả", self._save_all, style="outline").pack(fill="x")

        self.preview = PreviewPanel(self)
        self.preview.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    def _build_param_widgets(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        fv = self.filter_var.get()
        show_kernel = fv in ("mean", "gaussian")
        show_sigma  = fv == "gaussian"

        if not show_kernel:
            tk.Label(self.param_frame, text="Bộ lọc này không có tham số.",
                     font=FONT_SMALL, fg=TEXT_DIM, bg=SIDEBAR,
                     wraplength=220, justify="left").pack(anchor="w")
            return

        if show_kernel:
            row = tk.Frame(self.param_frame, bg=SIDEBAR)
            row.pack(fill="x", pady=3)
            tk.Label(row, text="Kernel size:", font=FONT_NORMAL,
                     fg=TEXT_MED, bg=SIDEBAR, width=12, anchor="w").pack(side="left")
            tk.Spinbox(row, from_=3, to=51, increment=2,
                       textvariable=self.kernel_var,
                       bg=CARD, fg=TEXT, font=FONT_NORMAL,
                       relief="solid", bd=1, width=5).pack(side="left")

        if show_sigma:
            row = tk.Frame(self.param_frame, bg=SIDEBAR)
            row.pack(fill="x", pady=3)
            tk.Label(row, text="Sigma:", font=FONT_NORMAL,
                     fg=TEXT_MED, bg=SIDEBAR, width=12, anchor="w").pack(side="left")
            
            def _sync_k(*_):
                try:
                    s = float(self.sigma_var.get())
                    k_rec = max(3, int(2 * np.ceil(2 * s) + 1)) | 1
                    if k_rec > self.kernel_var.get():
                        self.kernel_var.set(k_rec)
                except Exception:
                    pass

            self.sigma_var.trace_add("write", _sync_k)

            tk.Spinbox(row, from_=0.1, to=30.0, increment=0.5, format="%.1f",
                       textvariable=self.sigma_var,
                       bg=CARD, fg=TEXT, font=FONT_NORMAL,
                       relief="solid", bd=1, width=5).pack(side="left")

    def _on_filter_change(self):
        self._build_param_widgets()
        if self._cur_path and os.path.exists(self._cur_path):
            self._run()

    def _on_input(self, paths):
        self.load_paths(paths)
        if self.app:
            self.app.sync_input_paths(paths, sender_tab=self)

    def load_paths(self, paths):
        self.io.set_paths(paths)
        self._img_paths = expand_paths(paths)
        self.img_list.set_items([os.path.basename(p) for p in self._img_paths])
        if self._img_paths:
            self.img_list.select(0)
            self._cur_path = self._img_paths[0]
            self._run()

    def _on_select(self, idx):
        if idx < len(self._img_paths):
            self._cur_path = self._img_paths[idx]
            self._run()

    def _run(self):
        if not self._cur_path or not os.path.exists(self._cur_path):
            return

        # Đọc tham số trên luồng chính (Main thread) để đảm bảo an toàn thread và lấy giá trị mới nhất
        fv = self.filter_var.get()
        try:
            ksize = int(self.kernel_var.get())
            if ksize % 2 == 0:
                ksize += 1
        except Exception:
            ksize = 3

        try:
            sigma = float(self.sigma_var.get())
        except Exception:
            sigma = 1.0

        cur_path = self._cur_path

        def _task():
            try:
                self.after(0, lambda: self.status.info("Đang lọc ảnh..."))
                img_rgb  = load_image(cur_path)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

                if fv == "mean":
                    result, orig = self.filter_tool.mean_filter(img_rgb, kernel_size=ksize), img_rgb
                elif fv == "gaussian":
                    result, orig = self.filter_tool.gaussian_filter(img_rgb, kernel_size=ksize, sigma=sigma), img_rgb
                elif fv == "sobel_x":
                    result, orig = self.filter_tool.sobel_filter(img_gray, axis="x"), img_rgb
                elif fv == "sobel_y":
                    result, orig = self.filter_tool.sobel_filter(img_gray, axis="y"), img_rgb
                elif fv == "sobel_both":
                    result, orig = self.filter_tool.sobel_filter(img_gray, axis="both"), img_rgb
                elif fv == "laplacian_4":
                    result, orig = self.filter_tool.laplacian_filter(img_gray, connectivity=4), img_rgb
                elif fv == "laplacian_8":
                    result, orig = self.filter_tool.laplacian_filter(img_gray, connectivity=8), img_rgb
                else:
                    return

                self._cur_orig   = orig
                self._cur_result = result
                self.after(0, lambda: self.preview.set_images(orig, result))
                self.after(0, lambda: self.status.ok(
                    f"[{fv}] k={ksize}, sigma={sigma} — {os.path.basename(cur_path)}"))
            except Exception as e:
                self.after(0, lambda err=e: self.status.err(str(err)))
        threading.Thread(target=_task, daemon=True).start()

    def _save(self):
        if self._cur_result is None:
            messagebox.showwarning("Chưa có kết quả", "Chạy lọc ảnh trước!")
            return
        try:
            path = save_result(self._cur_orig, self._cur_result,
                               self.io.get_output_dir(), self.io.get_save_mode(),
                               self._cur_path or "filter_output.jpg")
            self.status.ok(f"Đã lưu → {path}")
            messagebox.showinfo("Đã lưu", f"Kết quả:\n{path}")
        except Exception as e:
            self.status.err(str(e))

    def _save_all(self):
        if not self._img_paths:
            messagebox.showwarning("Chưa có danh sách", "Chọn ảnh / thư mục trước!")
            return
        fv       = self.filter_var.get()
        ksize    = max(3, int(self.kernel_var.get()) | 1)
        sigma    = float(self.sigma_var.get())
        out_dir  = self.io.get_output_dir()
        save_mode= self.io.get_save_mode()
        total    = len(self._img_paths)

        def _task():
            count = 0
            try:
                for i, path in enumerate(self._img_paths):
                    self.after(0, lambda p=path, n=i: self.status.info(
                        f"[{n+1}/{total}] {os.path.basename(p)}"))
                    img_rgb  = load_image(path)
                    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    if fv == "mean":
                        res, orig = self.filter_tool.mean_filter(img_rgb, kernel_size=ksize), img_rgb
                    elif fv == "gaussian":
                        res, orig = self.filter_tool.gaussian_filter(img_rgb, ksize, sigma), img_rgb
                    elif fv == "sobel_x":
                        res, orig = self.filter_tool.sobel_filter(img_gray, axis="x"), img_rgb
                    elif fv == "sobel_y":
                        res, orig = self.filter_tool.sobel_filter(img_gray, axis="y"), img_rgb
                    elif fv == "sobel_both":
                        res, orig = self.filter_tool.sobel_filter(img_gray, axis="both"), img_rgb
                    elif fv == "laplacian_4":
                        res, orig = self.filter_tool.laplacian_filter(img_gray, connectivity=4), img_rgb
                    elif fv == "laplacian_8":
                        res, orig = self.filter_tool.laplacian_filter(img_gray, connectivity=8), img_rgb
                    else:
                        continue
                    save_result(orig, res, out_dir, save_mode, path)
                    count += 1
                self.after(0, lambda: self.status.ok(f"Đã lưu {count}/{total} ảnh → {out_dir}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Hoàn thành", f"Đã xử lý & lưu {count} ảnh vào:\n{out_dir}"))
            except Exception as e:
                self.after(0, lambda err=e: self.status.err(str(err)))
        threading.Thread(target=_task, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
class TransformTab(tk.Frame):
    def __init__(self, parent, status, app=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.status = status
        self.app = app
        self.transform = Transform()
        self._cur_orig = None
        self._cur_result = None
        self._cur_path = ""
        self._img_paths = []

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = ScrollableSidebar(self, width=295, bg=SIDEBAR)
        sidebar.pack(side="left", fill="y")
        sb = sidebar.content

        tk.Label(sb, text="🔄  Biến đổi Hình học",
                 font=FONT_TITLE, fg=ACCENT, bg=SIDEBAR).pack(pady=(14, 2), padx=14, anchor="w")
        tk.Label(sb, text="Translation · Rotation · Scaling · Affine · Projective",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=SIDEBAR, wraplength=260).pack(padx=14, anchor="w")
        hsep(sb)

        self.io = IOSection(sb, self._on_input, bg_color=SIDEBAR)
        self.io.pack(fill="x")
        hsep(sb)

        section_header(sb, "Loại biến đổi").pack(fill="x", padx=12, pady=(4, 6))
        self.trans_var = tk.StringVar(value="translation")
        self.trans_var.trace_add("write", lambda *_: self._update_param_ui())
        trans_opts = [
            ("Translation (Dịch chuyển)", "translation"),
            ("Rotation v1 (Forward)",     "rotation"),
            ("Rotation v2 (Inverse)",     "rotation_v2"),
            ("Scaling (Thu phóng)",       "scaling"),
            ("Affine Transformation",     "affine"),
            ("Projective Transformation", "projective")
        ]
        OptionButtonGroup(sb, trans_opts, self.trans_var, bg_color=SIDEBAR).pack(fill="x", padx=12, pady=(0, 4))
        hsep(sb)

        section_header(sb, "Tham số").pack(fill="x", padx=12, pady=(4, 4))
        self.param_frame = tk.Frame(sb, bg=SIDEBAR)
        self.param_frame.pack(fill="x", padx=14, pady=(0, 6))
        self._params = {}
        self._update_param_ui()
        hsep(sb)

        section_header(sb, "Danh sách ảnh").pack(fill="x", padx=12, pady=(4, 4))
        self.img_list = ImageListBox(sb, self._on_select, height=4)
        self.img_list.pack(fill="x", padx=12, pady=(0, 6))

        btn_r = tk.Frame(sb, bg=SIDEBAR)
        btn_r.pack(fill="x", padx=12, pady=(0, 10))

        row1 = tk.Frame(btn_r, bg=SIDEBAR)
        row1.pack(fill="x", pady=(0, 4))
        make_btn(row1, "▶  Biến đổi", self._run, style="primary").pack(side="left", expand=True, fill="x", padx=(0, 4))
        make_btn(row1, "💾  Lưu ảnh này", self._save, style="success").pack(side="left", expand=True, fill="x")

        make_btn(btn_r, "⚡  Biến đổi & Lưu tất cả", self._save_all, style="outline").pack(fill="x")

        self.preview = PreviewPanel(self)
        self.preview.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    def _add_param(self, label, key, typ=tk.DoubleVar, init=0.0):
        row = tk.Frame(self.param_frame, bg=SIDEBAR)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=f"{label}:", font=FONT_NORMAL,
                 fg=TEXT_MED, bg=SIDEBAR, width=13, anchor="w").pack(side="left")
        var = typ(value=init)
        self._params[key] = var
        kw = dict(textvariable=var, bg=CARD, fg=TEXT, font=FONT_NORMAL,
                  relief="solid", bd=1, width=8)
        if typ == tk.IntVar:
            tk.Spinbox(row, from_=-9999, to=9999, **kw).pack(side="left")
        else:
            tk.Spinbox(row, from_=-9999.0, to=9999.0,
                       increment=0.5, format="%.2f", **kw).pack(side="left")

    def _update_param_ui(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self._params.clear()
        tv = self.trans_var.get()
        if tv == "translation":
            self._add_param("tx (pixels)", "tx", tk.IntVar, 20)
            self._add_param("ty (pixels)", "ty", tk.IntVar, 20)
        elif tv in ("rotation", "rotation_v2"):
            self._add_param("Góc (độ)", "angle", tk.DoubleVar, 45.0)
        elif tv == "scaling":
            self._add_param("Scale X", "scale_x", tk.DoubleVar, 0.75)
            self._add_param("Scale Y", "scale_y", tk.DoubleVar, 0.75)
        else:  # affine / projective
            tk.Label(self.param_frame, text="Dùng preset mặc định:\ngóc ảnh offset 50px.",
                     font=FONT_SMALL, fg=TEXT_DIM, bg=SIDEBAR, justify="left").pack(anchor="w")
        if self._cur_path and os.path.exists(self._cur_path):
            self.after(60, self._run)

    def _on_input(self, paths):
        self.load_paths(paths)
        if self.app:
            self.app.sync_input_paths(paths, sender_tab=self)

    def load_paths(self, paths):
        self.io.set_paths(paths)
        self._img_paths = expand_paths(paths)
        self.img_list.set_items([os.path.basename(p) for p in self._img_paths])
        if self._img_paths:
            self.img_list.select(0)
            self._cur_path = self._img_paths[0]
            self._run()

    def _on_select(self, idx):
        if idx < len(self._img_paths):
            self._cur_path = self._img_paths[idx]
            self._run()

    def _run(self):
        if not self._cur_path or not os.path.exists(self._cur_path):
            return

        def _task():
            try:
                self.after(0, lambda: self.status.info("Đang biến đổi ảnh..."))
                img  = load_image(self._cur_path)
                rows, cols = img.shape[:2]
                tv   = self.trans_var.get()

                if tv == "translation":
                    tx = int(self._params["tx"].get()) if "tx" in self._params else 20
                    ty = int(self._params["ty"].get()) if "ty" in self._params else 20
                    result = self.transform.translation(img, tx=tx, ty=ty)
                elif tv == "rotation":
                    angle = float(self._params["angle"].get()) if "angle" in self._params else 45.0
                    result = self.transform.rotation(img, angle=angle)
                elif tv == "rotation_v2":
                    angle = float(self._params["angle"].get()) if "angle" in self._params else 45.0
                    result = self.transform.rotation_v2(img, angle=angle)
                elif tv == "scaling":
                    sx = float(self._params["scale_x"].get()) if "scale_x" in self._params else 0.75
                    sy = float(self._params["scale_y"].get()) if "scale_y" in self._params else 0.75
                    result = self.transform.scaling(img, scale_x=sx, scale_y=sy)
                elif tv == "affine":
                    src = np.float32([[0,0],[cols-1,0],[0,rows-1]])
                    dst = np.float32([[0,0],[cols-50,0],[50,rows-1]])
                    result = self.transform.affine(img, src, dst)
                elif tv == "projective":
                    src = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
                    dst = np.float32([[0,0],[cols-1,0],[50,rows-1],[cols-50,rows-1]])
                    result = self.transform.projective(img, src, dst)
                else:
                    return

                self._cur_orig   = img
                self._cur_result = result
                self.after(0, lambda: self.preview.set_images(img, result))
                self.after(0, lambda: self.status.ok(
                    f"[{tv}] — {os.path.basename(self._cur_path)}"))
            except Exception as e:
                self.after(0, lambda err=e: self.status.err(str(err)))
        threading.Thread(target=_task, daemon=True).start()

    def _save(self):
        if self._cur_result is None:
            messagebox.showwarning("Chưa có kết quả", "Chạy biến đổi trước!")
            return
        try:
            path = save_result(self._cur_orig, self._cur_result,
                               self.io.get_output_dir(), self.io.get_save_mode(),
                               self._cur_path or "transform_output.jpg")
            self.status.ok(f"Đã lưu → {path}")
            messagebox.showinfo("Đã lưu", f"Kết quả:\n{path}")
        except Exception as e:
            self.status.err(str(e))

    def _save_all(self):
        if not self._img_paths:
            messagebox.showwarning("Chưa có danh sách", "Chọn ảnh / thư mục trước!")
            return
        tv        = self.trans_var.get()
        out_dir   = self.io.get_output_dir()
        save_mode = self.io.get_save_mode()
        total     = len(self._img_paths)
        params_snap = {k: v.get() for k, v in self._params.items()}

        def _task():
            count = 0
            try:
                for i, path in enumerate(self._img_paths):
                    self.after(0, lambda p=path, n=i: self.status.info(
                        f"[{n+1}/{total}] {os.path.basename(p)}"))
                    img  = load_image(path)
                    rows, cols = img.shape[:2]
                    if tv == "translation":
                        result = self.transform.translation(
                            img, tx=int(params_snap.get("tx", 20)),
                            ty=int(params_snap.get("ty", 20)))
                    elif tv == "rotation":
                        result = self.transform.rotation(
                            img, angle=float(params_snap.get("angle", 45)))
                    elif tv == "rotation_v2":
                        result = self.transform.rotation_v2(
                            img, angle=float(params_snap.get("angle", 45)))
                    elif tv == "scaling":
                        result = self.transform.scaling(
                            img, scale_x=float(params_snap.get("scale_x", 0.75)),
                            scale_y=float(params_snap.get("scale_y", 0.75)))
                    elif tv == "affine":
                        src = np.float32([[0,0],[cols-1,0],[0,rows-1]])
                        dst = np.float32([[0,0],[cols-50,0],[50,rows-1]])
                        result = self.transform.affine(img, src, dst)
                    elif tv == "projective":
                        src = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
                        dst = np.float32([[0,0],[cols-1,0],[50,rows-1],[cols-50,rows-1]])
                        result = self.transform.projective(img, src, dst)
                    else:
                        continue
                    save_result(img, result, out_dir, save_mode, path)
                    count += 1
                self.after(0, lambda: self.status.ok(f"Đã lưu {count}/{total} ảnh → {out_dir}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Hoàn thành", f"Đã xử lý & lưu {count} ảnh vào:\n{out_dir}"))
            except Exception as e:
                self.after(0, lambda err=e: self.status.err(str(err)))
        threading.Thread(target=_task, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Computer Vision Lab")
        self.geometry("1340x820")
        self.minsize(960, 640)
        self.configure(bg=BG)

        # Header
        header = tk.Frame(self, bg=ACCENT, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="  ⬡  Computer Vision Lab",
                 font=("Segoe UI", 13, "bold"), fg="white",
                 bg=ACCENT).pack(side="left", padx=16, pady=10)
        tk.Label(header, text="BTL – HCMUT  ",
                 font=FONT_SMALL, fg="#C5CAE9",
                 bg=ACCENT).pack(side="right", padx=16)

        # Notebook
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",
                         background=ACCENT, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab",
                         background=ACCENT_DARK, foreground="#C5CAE9",
                         font=FONT_BTN, padding=[18, 8], borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG), ("active", ACCENT_LIGHT)],
                  foreground=[("selected", ACCENT), ("active", ACCENT)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.status = StatusBar(self)
        self.status.pack(fill="x", side="bottom")

        self._tab1 = ColorTab(nb, self.status, app=self)
        self._tab2 = FilterTab(nb, self.status, app=self)
        self._tab3 = TransformTab(nb, self.status, app=self)

        nb.add(self._tab1, text="  🎨  Màu & Kênh màu  ")
        nb.add(self._tab2, text="  🔍  Lọc ảnh (Filter)  ")
        nb.add(self._tab3, text="  🔄  Biến đổi hình học  ")

    def sync_input_paths(self, paths, sender_tab=None):
        """Broadcast danh sách ảnh sang các tab còn lại (không re-trigger sender)."""
        for tab in (self._tab1, self._tab2, self._tab3):
            if tab is not sender_tab:
                tab.load_paths(paths)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
