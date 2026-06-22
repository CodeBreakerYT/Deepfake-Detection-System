import customtkinter as ctk

from desktop import theme
from desktop.widgets import HeuristicsPanel, TimelineCanvas
from desktop.pages.base import BaseAnalysisPage


class VoicePage(BaseAnalysisPage):
    def __init__(self, master, app):
        self.app = app
        self.current_report = None
        super().__init__(
            master,
            page_title="Voice Deepfake Analysis",
            page_subtitle="Detect AI-generated speech, voice cloning, and synthetic vocoder artifacts in audio recordings.",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac")],
        )

        self.heuristics_panel = HeuristicsPanel(self.results_view, "Acoustic Artifact Markers")
        self.heuristics_panel.pack(fill="x", pady=(0, 16))

        timeline_card = ctk.CTkFrame(self.results_view, fg_color=theme.BG_CARD, corner_radius=14)
        timeline_card.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(timeline_card, text="Per-Segment Score Timeline (click a point to inspect that segment)",
                     font=(theme.FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=16, pady=(14, 4))
        self.timeline = TimelineCanvas(timeline_card, on_select=self._on_segment_selected)
        self.timeline.pack(fill="x", padx=16, pady=(0, 16))

        self.segment_card = ctk.CTkFrame(self.results_view, fg_color=theme.BG_CARD, corner_radius=14)
        self.segment_card.pack(fill="x", pady=(0, 16))
        self.segment_title = ctk.CTkLabel(self.segment_card, text="", font=(theme.FONT_FAMILY, 13, "bold"))
        self.segment_title.pack(anchor="w", padx=16, pady=(14, 4))
        self.segment_detail = ctk.CTkLabel(self.segment_card, text="", font=(theme.FONT_FAMILY, 12),
                                            text_color=theme.TEXT_SECONDARY, justify="left")
        self.segment_detail.pack(anchor="w", padx=16, pady=(0, 16))

        ctk.CTkButton(self.results_view, text="Analyze Another Recording", command=self.reset,
                      fg_color=theme.BG_CARD_HOVER, hover_color=theme.BORDER).pack(anchor="w")

    def run_analysis(self, path, on_progress):
        file_hash = self.app.voice_pipeline._get_file_hash(path)
        cached = self.app.store.get_cached_result(file_hash)
        if cached:
            return cached
        report = self.app.voice_pipeline.analyze_audio(path, update_progress_cb=on_progress)
        self.app.store.cache_result(file_hash, report)
        return report

    def render_result(self, report: dict):
        self.current_report = report
        mode_line = ("Transformer Model Active (wav2vec2-deepfake-voice-detector)" if report.get("used_voice_model")
                     else "Local Acoustic Heuristics Active")
        self.gauge_card.render(
            filename=report["filename"],
            mode_line=mode_line,
            fake_score=report["global_fake_score"],
            is_fake=report["is_fake"],
            stats=[
                ("Analysis Target", "Audio File"),
                ("Processing Latency", f"{report['processing_time_sec']}s"),
                ("Clip Duration", f"{report['duration_sec']}s"),
                ("Segments Analyzed", report["total_segments_analyzed"]),
            ],
        )
        self.heuristics_panel.render([
            ("Pitch Jitter Anomaly", report["average_heuristics"]["pitch_jitter_score"]),
            ("Spectral Flatness Anomaly", report["average_heuristics"]["spectral_flatness_score"]),
            ("Energy/Silence Anomaly", report["average_heuristics"]["silence_anomaly_score"]),
        ])

        points = [(s["segment_idx"], s["fake_score"]) for s in report.get("segments", [])]
        self.timeline.set_points(points, selected_idx=points[0][0] if points else 0)

        if points:
            self._show_segment(points[0][0])

    def _on_segment_selected(self, segment_idx: int):
        self._show_segment(segment_idx)

    def _show_segment(self, segment_idx: int):
        if not self.current_report:
            return
        seg = next((s for s in self.current_report["segments"] if s["segment_idx"] == segment_idx), None)
        if not seg:
            return
        self.segment_title.configure(
            text=f"Segment #{segment_idx + 1}  ({seg['start_time']}s – {seg['end_time']}s)  "
                 f"— {round(seg['fake_score'] * 100)}% fake"
        )
        h = seg["heuristics"]
        self.segment_detail.configure(
            text=f"Pitch Jitter Value: {h['pitch_jitter_val']}\n"
                 f"Spectral Flatness Value: {h['spectral_flatness_val']}\n"
                 f"Energy Variation (CV): {h['energy_variation_val']}"
        )
