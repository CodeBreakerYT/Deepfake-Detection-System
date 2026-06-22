import customtkinter as ctk

from desktop import theme


class AboutPage(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")

        ctk.CTkLabel(self, text="About DeepShield", font=(theme.FONT_FAMILY, 22, "bold")).pack(anchor="w")
        ctk.CTkLabel(self, text="How the detection pipeline works, and what it can and can't tell you.",
                     font=(theme.FONT_FAMILY, 12), text_color=theme.TEXT_SECONDARY).pack(anchor="w", pady=(2, 20))

        self._section(
            "Detection Architecture",
            "Every file runs through a two-layer pipeline. First, a deep learning classifier scores the "
            "media: a fine-tuned Vision Transformer (Wvolf/ViT_Deepfake_Detection) for faces in images and "
            "video frames, and a fine-tuned Wav2Vec2 model (wav2vec2-deepfake-voice-detector) for speech "
            "audio. Second, classical forensic heuristics run alongside the model and act as a fallback if "
            "a model is unavailable: Laplacian blur variance, FFT spectral-ratio anomalies, and color-channel "
            "deviation for visual media; pitch jitter, spectral flatness, and energy-variation analysis for audio.",
        )
        self._section(
            "Image & Video",
            "Faces are located with OpenCV Haar cascades and cropped before classification. Videos are "
            "sampled at up to 30 evenly-spaced frames; every face per frame is scored independently. The "
            "global score blends the single worst face score (70%) with the average across all faces (30%) "
            "— one convincing fake face is enough to flag a video.",
        )
        self._section(
            "Voice",
            "Audio is resampled to 16kHz mono and split into overlapping 3-second windows. Each window is "
            "scored independently, producing a per-segment timeline similar to video frames. Acoustic "
            "heuristics target known synthetic-speech tells: unnaturally stable or erratic pitch, flat "
            "vocoder-like spectra, and uniform energy envelopes.",
        )
        self._section(
            "Limitations & Disclaimer",
            "No deepfake detector is perfect — scores are probabilistic signals, not legal proof. Heavily "
            "compressed, low-resolution, or adversarially-processed media can fool any model. Treat results "
            "as one input among several when verifying media authenticity, especially for high-stakes decisions.",
            color=theme.WARNING,
        )
        self._section(
            "Runs Fully Offline",
            "This desktop build does not require an account or a network connection to analyze files (only "
            "the one-time model download needs internet). Nothing is uploaded anywhere — all processing "
            "happens on your machine, and results are cached locally in deepshield_history.json.",
        )

    def _section(self, title, body, color=None):
        card = ctk.CTkFrame(self, fg_color=theme.BG_CARD, corner_radius=14,
                             border_width=1 if color else 0, border_color=color or theme.BORDER)
        card.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(card, text=title, font=(theme.FONT_FAMILY, 14, "bold"),
                     text_color=color or theme.TEXT_PRIMARY).pack(anchor="w", padx=18, pady=(16, 6))
        ctk.CTkLabel(card, text=body, font=(theme.FONT_FAMILY, 12), text_color=theme.TEXT_SECONDARY,
                     wraplength=780, justify="left").pack(anchor="w", padx=18, pady=(0, 16))


class AboutScreen(ctk.CTkFrame):
    """Full-screen About view reachable from the start screen, with a back button."""

    def __init__(self, master, on_back):
        super().__init__(master, fg_color=theme.BG_MAIN)
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 0))
        ctk.CTkButton(
            top, text="← Back", width=90, height=32, fg_color=theme.BG_CARD_HOVER,
            hover_color=theme.BORDER, command=on_back
        ).pack(anchor="w")

        body = AboutPage(self)
        body.pack(fill="both", expand=True, padx=24, pady=16)
