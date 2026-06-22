import customtkinter as ctk

from desktop import theme
from desktop.animations import pulse_widget


class StartPage(ctk.CTkFrame):
    def __init__(self, master, on_start, on_quit, on_about):
        super().__init__(master, fg_color=theme.BG_MAIN)
        self._pulse_cancel = None

        center = ctk.CTkFrame(self, fg_color="transparent")
        center.place(relx=0.5, rely=0.46, anchor="center")

        badge = ctk.CTkLabel(
            center, text="🛡", font=(theme.FONT_FAMILY, 46), text_color=theme.PRIMARY
        )
        badge.pack(pady=(0, 10))

        ctk.CTkLabel(
            center, text="Deepfake Detection System",
            font=(theme.FONT_FAMILY, 30, "bold")
        ).pack()

        ctk.CTkLabel(
            center,
            text="Verify images, video, and voice recordings for AI-generated forgeries — fully offline.",
            font=(theme.FONT_FAMILY, 13), text_color=theme.TEXT_SECONDARY
        ).pack(pady=(8, 30))

        self.status_label = ctk.CTkLabel(
            center, text="● Loading detection models...", font=(theme.FONT_FAMILY, 11),
            text_color=theme.WARNING
        )
        self.status_label.pack(pady=(0, 18))

        btn_row = ctk.CTkFrame(center, fg_color="transparent")
        btn_row.pack()

        self.start_btn = ctk.CTkButton(
            btn_row, text="Loading...", width=160, height=44, font=(theme.FONT_FAMILY, 14, "bold"),
            fg_color=theme.PRIMARY, hover_color=theme.SECONDARY, corner_radius=10,
            state="disabled", border_width=2, border_color=theme.PRIMARY,
            command=on_start,
        )
        self.start_btn.pack(side="left", padx=8)

        ctk.CTkButton(
            btn_row, text="Quit", width=120, height=44, font=(theme.FONT_FAMILY, 14, "bold"),
            fg_color=theme.BG_CARD_HOVER, hover_color=theme.DANGER, corner_radius=10,
            command=on_quit,
        ).pack(side="left", padx=8)

        about_link = ctk.CTkLabel(
            self, text="About this tool", font=(theme.FONT_FAMILY, 11, "underline"),
            text_color=theme.TEXT_MUTED, cursor="hand2"
        )
        about_link.place(relx=0.5, rely=0.92, anchor="center")
        about_link.bind("<Button-1>", lambda e: on_about())

    def set_ready(self):
        self.start_btn.configure(state="normal", text="Start")
        self.status_label.configure(text="● Ready", text_color=theme.SUCCESS)
        self._pulse_cancel = pulse_widget(self.start_btn, theme.PRIMARY, theme.SECONDARY,
                                           prop="border_color", interval=900)

    def destroy(self):
        if self._pulse_cancel:
            self._pulse_cancel()
        super().destroy()
