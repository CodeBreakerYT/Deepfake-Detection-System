import os
import queue
import threading
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk

from desktop import theme
from desktop.widgets import ProgressPanel, GaugeCard, HeuristicsPanel


class BaseAnalysisPage(ctk.CTkFrame):
    """
    Shared upload -> background-thread analysis -> progress -> results/error
    state machine used by the Image, Video, and Voices pages. Subclasses
    provide the file dialog filter, the analysis function, and how to render
    the type-specific parts of the result (face gallery, timeline, etc).
    """

    def __init__(self, master, page_title: str, page_subtitle: str, filetypes: list[tuple[str, str]]):
        super().__init__(master, fg_color="transparent")
        self.filetypes = filetypes
        self._queue: queue.Queue = queue.Queue()
        self._current_filename = ""

        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        ctk.CTkLabel(header, text=page_title, font=(theme.FONT_FAMILY, 22, "bold")).pack(anchor="w")
        ctk.CTkLabel(
            header, text=page_subtitle, font=(theme.FONT_FAMILY, 12),
            text_color=theme.TEXT_SECONDARY, wraplength=700, justify="left"
        ).pack(anchor="w", pady=(4, 0))

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)

        self.upload_view = self._build_upload_view()
        self.progress_view = ProgressPanel(self.body)
        self.error_view = self._build_error_view()
        self.results_view = ctk.CTkScrollableFrame(self.body, fg_color="transparent")

        self.gauge_card = GaugeCard(self.results_view)
        self.gauge_card.pack(fill="x", pady=(0, 16))
        self.heuristics_panel = None  # set by subclass title

        self._show(self.upload_view)

    # ---- views ----

    def _build_upload_view(self):
        frame = ctk.CTkFrame(self.body, fg_color=theme.BG_CARD, corner_radius=14, border_width=2,
                              border_color=theme.BORDER)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner, text="↑", font=(theme.FONT_FAMILY, 32), text_color=theme.PRIMARY).pack()
        ctk.CTkLabel(inner, text="Click to choose a file", font=(theme.FONT_FAMILY, 16, "bold")).pack(pady=(8, 4))
        btn = ctk.CTkButton(inner, text="Browse...", command=self._choose_file, width=160,
                             fg_color=theme.PRIMARY, hover_color=theme.SECONDARY)
        btn.pack(pady=(8, 0))
        frame.configure(height=260)
        return frame

    def _build_error_view(self):
        frame = ctk.CTkFrame(self.body, fg_color=theme.BG_CARD, corner_radius=14, border_width=1,
                              border_color=theme.DANGER)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        self.error_label = ctk.CTkLabel(inner, text="", font=(theme.FONT_FAMILY, 13),
                                         text_color=theme.TEXT_SECONDARY, wraplength=500)
        self.error_label.pack(pady=(0, 12))
        ctk.CTkButton(inner, text="Try Again", command=self.reset, fg_color=theme.PRIMARY).pack()
        frame.configure(height=200)
        return frame

    def _show(self, view):
        for child in self.body.winfo_children():
            child.place_forget()
            child.pack_forget()
        view.pack(fill="both", expand=True)

    # ---- lifecycle ----

    def reset(self):
        self._show(self.upload_view)

    def _choose_file(self):
        path = filedialog.askopenfilename(filetypes=self.filetypes)
        if not path:
            return
        self._current_filename = os.path.basename(path)
        self._show(self.progress_view)
        self.progress_view.update_progress(5, "Starting analysis...", self._current_filename)
        threading.Thread(target=self._worker, args=(path,), daemon=True).start()
        self.after(100, self._poll_queue)

    def _worker(self, path: str):
        try:
            def on_progress(pct, stage):
                self._queue.put(("progress", pct, stage))

            report = self.run_analysis(path, on_progress)
            self._queue.put(("result", report))
        except Exception as e:
            traceback.print_exc()
            self._queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, pct, stage = item
                    self.progress_view.update_progress(pct, stage, self._current_filename)
                elif kind == "result":
                    self._show(self.results_view)
                    self.render_result(item[1])
                    return
                elif kind == "error":
                    self.error_label.configure(text=item[1])
                    self._show(self.error_view)
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ---- to override ----

    def run_analysis(self, path: str, on_progress):
        raise NotImplementedError

    def render_result(self, report: dict):
        raise NotImplementedError
