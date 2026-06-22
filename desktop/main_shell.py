import customtkinter as ctk

from desktop import theme
from desktop.animations import slide_in
from desktop.pages.image_page import ImagePage
from desktop.pages.video_page import VideoPage
from desktop.pages.voice_page import VoicePage

TAB_ORDER = ["Image", "Video", "Voice"]


class MainShell(ctk.CTkFrame):
    """Top bar (title + Image/Video/Voice tabs + Quit) with an animated content area below."""

    def __init__(self, master, app, on_quit, on_home):
        super().__init__(master, fg_color=theme.BG_MAIN)

        topbar = ctk.CTkFrame(self, fg_color=theme.BG_CARD, height=64, corner_radius=0)
        topbar.pack(fill="x", side="top")

        left = ctk.CTkFrame(topbar, fg_color="transparent")
        left.pack(side="left", padx=20, pady=12)
        ctk.CTkButton(
            left, text="⌂", width=36, height=36, font=(theme.FONT_FAMILY, 15),
            fg_color="transparent", hover_color=theme.BG_CARD_HOVER, command=on_home
        ).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(
            left, text="Deepfake Detection System", font=(theme.FONT_FAMILY, 15, "bold")
        ).pack(side="left")

        self.segmented = ctk.CTkSegmentedButton(
            topbar, values=TAB_ORDER, command=self._on_tab_change,
            font=(theme.FONT_FAMILY, 13, "bold"), height=36, width=320,
            selected_color=theme.PRIMARY, selected_hover_color=theme.SECONDARY,
            unselected_color=theme.BG_CARD_HOVER, unselected_hover_color=theme.BORDER,
        )
        self.segmented.place(relx=0.5, rely=0.5, anchor="center")

        right = ctk.CTkFrame(topbar, fg_color="transparent")
        right.pack(side="right", padx=20, pady=12)
        ctk.CTkButton(
            right, text="Quit", width=90, height=36, font=(theme.FONT_FAMILY, 13, "bold"),
            fg_color=theme.BG_CARD_HOVER, hover_color=theme.DANGER, command=on_quit
        ).pack()

        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.pack(fill="both", expand=True, padx=24, pady=20)

        self.app = app
        self.pages = {}
        self.current_key = None

    def build_pages(self):
        if self.pages:
            return
        self.pages["Image"] = ImagePage(self.content_area, self.app)
        self.pages["Video"] = VideoPage(self.content_area, self.app)
        self.pages["Voice"] = VoicePage(self.content_area, self.app)
        self.segmented.set("Image")
        self._show("Image", animate=False)

    def _on_tab_change(self, value: str):
        self._show(value, animate=True)

    def _show(self, key: str, animate: bool = True):
        if key not in self.pages or key == self.current_key:
            return

        prev_idx = TAB_ORDER.index(self.current_key) if self.current_key in TAB_ORDER else -1
        next_idx = TAB_ORDER.index(key)
        direction = "right" if next_idx > prev_idx else "left"

        for k, p in self.pages.items():
            if k != key:
                p.place_forget()

        target = self.pages[key]
        if animate:
            slide_in(target, self.content_area, direction=direction)
        else:
            target.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.current_key = key
