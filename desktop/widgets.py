import base64
import io
import tkinter as tk
import customtkinter as ctk
from PIL import Image

from desktop import theme


def format_pct(val: float) -> str:
    return f"{round(val * 100)}"


class GaugeCard(ctk.CTkFrame):
    """Big circular-feeling score readout with a verdict badge, plus a stats panel."""

    def __init__(self, master):
        super().__init__(master, fg_color=theme.BG_CARD, corner_radius=14)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.score_panel = ctk.CTkFrame(self, fg_color="transparent", width=220)
        self.score_panel.grid(row=0, column=0, padx=20, pady=20, sticky="n")

        self.score_label = ctk.CTkLabel(
            self.score_panel, text="--%", font=(theme.FONT_FAMILY, 38, "bold")
        )
        self.score_label.pack(pady=(10, 0))
        self.score_sub = ctk.CTkLabel(
            self.score_panel, text="FAKE SCORE", font=(theme.FONT_FAMILY, 11),
            text_color=theme.TEXT_SECONDARY
        )
        self.score_sub.pack()
        self.bar = ctk.CTkProgressBar(self.score_panel, width=180, height=10)
        self.bar.pack(pady=(14, 8))
        self.bar.set(0)

        self.verdict_label = ctk.CTkLabel(
            self.score_panel, text="", font=(theme.FONT_FAMILY, 12, "bold"),
            corner_radius=999, fg_color=theme.BG_CARD_HOVER, padx=14, pady=4
        )
        self.verdict_label.pack(pady=(4, 10))

        self.stats_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(
            self.stats_panel, text="", font=(theme.FONT_FAMILY, 16, "bold"), anchor="w"
        )
        self.title_label.pack(fill="x")
        self.mode_label = ctk.CTkLabel(
            self.stats_panel, text="", font=(theme.FONT_FAMILY, 11),
            text_color=theme.TEXT_MUTED, anchor="w"
        )
        self.mode_label.pack(fill="x", pady=(2, 14))

        self.stats_grid = ctk.CTkFrame(self.stats_panel, fg_color="transparent")
        self.stats_grid.pack(fill="x")

    def render(self, filename: str, mode_line: str, fake_score: float, is_fake: bool, stats: list[tuple[str, str]]):
        self.score_label.configure(text=f"{format_pct(fake_score)}%")
        self.bar.set(fake_score)
        color = theme.DANGER if is_fake else theme.SUCCESS
        self.bar.configure(progress_color=color)
        self.verdict_label.configure(
            text="DEEPFAKE FLAGGED" if is_fake else "VERIFIED AUTHENTIC",
            text_color=color
        )
        self.title_label.configure(text=filename)
        self.mode_label.configure(text=mode_line)

        for widget in self.stats_grid.winfo_children():
            widget.destroy()

        for i, (label, value) in enumerate(stats):
            row, col = divmod(i, 2)
            cell = ctk.CTkFrame(self.stats_grid, fg_color=theme.BG_CARD_HOVER, corner_radius=10)
            cell.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self.stats_grid.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(
                cell, text=label.upper(), font=(theme.FONT_FAMILY, 9, "bold"),
                text_color=theme.TEXT_SECONDARY
            ).pack(anchor="w", padx=10, pady=(8, 0))
            ctk.CTkLabel(
                cell, text=str(value), font=(theme.FONT_FAMILY, 14, "bold")
            ).pack(anchor="w", padx=10, pady=(0, 8))


class HeuristicsPanel(ctk.CTkFrame):
    """Row of labeled progress bars for forensic heuristic scores."""

    def __init__(self, master, title: str):
        super().__init__(master, fg_color=theme.BG_CARD, corner_radius=14)
        ctk.CTkLabel(
            self, text=title, font=(theme.FONT_FAMILY, 14, "bold")
        ).pack(anchor="w", padx=18, pady=(16, 10))
        self.rows_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.rows_frame.pack(fill="x", padx=18, pady=(0, 16))

    def render(self, items: list[tuple[str, float]]):
        for widget in self.rows_frame.winfo_children():
            widget.destroy()

        for label, value in items:
            row = ctk.CTkFrame(self.rows_frame, fg_color="transparent")
            row.pack(fill="x", pady=6)

            header = ctk.CTkFrame(row, fg_color="transparent")
            header.pack(fill="x")
            ctk.CTkLabel(
                header, text=label, font=(theme.FONT_FAMILY, 12), text_color=theme.TEXT_SECONDARY
            ).pack(side="left")
            ctk.CTkLabel(
                header, text=f"{format_pct(value)}%", font=(theme.FONT_FAMILY, 12, "bold"),
                text_color=theme.heuristic_color(value)
            ).pack(side="right")

            bar = ctk.CTkProgressBar(row, height=8, progress_color=theme.heuristic_color(value))
            bar.pack(fill="x", pady=(4, 0))
            bar.set(value)


class TimelineCanvas(tk.Canvas):
    """Lightweight clickable score-over-time chart (frames or audio segments)."""

    def __init__(self, master, on_select=None, **kwargs):
        super().__init__(master, height=140, bg=theme.BG_CARD, highlightthickness=0, **kwargs)
        self.on_select = on_select
        self.points = []
        self.selected_idx = 0
        self.bind("<Configure>", lambda e: self._redraw())

    def set_points(self, points: list[tuple[int, float]], selected_idx: int = 0):
        self.points = points
        self.selected_idx = selected_idx
        self._redraw()

    def select(self, idx: int):
        self.selected_idx = idx
        self._redraw()

    def _redraw(self):
        self.delete("all")
        if len(self.points) < 2:
            return

        w = max(self.winfo_width(), 200)
        h = 140
        pad_x, pad_y = 20, 20
        plot_w = w - 2 * pad_x
        plot_h = h - 2 * pad_y

        coords = []
        for i, (idx, score) in enumerate(self.points):
            x = pad_x + (i / (len(self.points) - 1)) * plot_w
            y = pad_y + (1 - score) * plot_h
            coords.append((x, y, idx))

        for i in range(len(coords) - 1):
            self.create_line(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1],
                              fill=theme.SECONDARY, width=2, smooth=True)

        for x, y, idx in coords:
            is_selected = idx == self.selected_idx
            r = 6 if is_selected else 4
            color = theme.PRIMARY if not is_selected else theme.SECONDARY
            item = self.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white" if is_selected else "")
            self.tag_bind(item, "<Button-1>", lambda e, i=idx: self._handle_click(i))

    def _handle_click(self, idx: int):
        self.selected_idx = idx
        self._redraw()
        if self.on_select:
            self.on_select(idx)


class FaceGallery(ctk.CTkScrollableFrame):
    """Grid of face-crop thumbnails decoded from base64 JPEGs, with fake-score captions."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=theme.BG_CARD, corner_radius=14, height=220, **kwargs)
        self._thumb_refs = []

    def render(self, faces: list[dict], on_click=None):
        for widget in self.winfo_children():
            widget.destroy()
        self._thumb_refs = []

        cols = 6
        for i, face in enumerate(faces):
            row, col = divmod(i, cols)
            cell = ctk.CTkFrame(self, fg_color=theme.BG_CARD_HOVER, corner_radius=10)
            cell.grid(row=row, column=col, padx=6, pady=6)

            crop_b64 = face.get("crop_b64", "")
            if crop_b64:
                img_bytes = base64.b64decode(crop_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).resize((90, 90))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(90, 90))
                self._thumb_refs.append(ctk_img)
                lbl = ctk.CTkLabel(cell, image=ctk_img, text="")
                lbl.pack(padx=6, pady=(6, 2))
                if on_click:
                    lbl.bind("<Button-1>", lambda e, f=face: on_click(f))

            score = face.get("fake_score", 0.0)
            ctk.CTkLabel(
                cell, text=f"{format_pct(score)}% fake", font=(theme.FONT_FAMILY, 10, "bold"),
                text_color=theme.heuristic_color(score)
            ).pack(pady=(0, 6))


class ProgressPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=theme.BG_CARD, corner_radius=14)
        self.stage_label = ctk.CTkLabel(self, text="", font=(theme.FONT_FAMILY, 15, "bold"))
        self.stage_label.pack(pady=(24, 4))
        self.file_label = ctk.CTkLabel(self, text="", font=(theme.FONT_FAMILY, 11), text_color=theme.TEXT_MUTED)
        self.file_label.pack()
        self.bar = ctk.CTkProgressBar(self, width=360, height=10)
        self.bar.pack(pady=20)
        self.bar.set(0)
        self.pct_label = ctk.CTkLabel(self, text="0%", font=(theme.FONT_FAMILY, 22, "bold"))
        self.pct_label.pack(pady=(0, 24))

    def update_progress(self, pct: int, stage: str, filename: str):
        self.bar.set(pct / 100)
        self.pct_label.configure(text=f"{pct}%")
        self.stage_label.configure(text=stage)
        self.file_label.configure(text=filename)
