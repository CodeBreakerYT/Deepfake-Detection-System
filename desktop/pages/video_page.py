import customtkinter as ctk

from desktop import theme
from desktop.widgets import HeuristicsPanel, FaceGallery, TimelineCanvas
from desktop.pages.base import BaseAnalysisPage


class VideoPage(BaseAnalysisPage):
    def __init__(self, master, app):
        self.app = app
        self.current_report = None
        super().__init__(
            master,
            page_title="Video Deepfake Analysis",
            page_subtitle="Sample frames, track faces over time, and flag temporal inconsistencies frame by frame.",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm")],
        )

        self.heuristics_panel = HeuristicsPanel(self.results_view, "Pixel-Level Visual Artifact Markers")
        self.heuristics_panel.pack(fill="x", pady=(0, 16))

        timeline_card = ctk.CTkFrame(self.results_view, fg_color=theme.BG_CARD, corner_radius=14)
        timeline_card.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(timeline_card, text="Temporal Score Analysis (click a point to inspect that frame)",
                     font=(theme.FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=16, pady=(14, 4))
        self.timeline = TimelineCanvas(timeline_card, on_select=self._on_frame_selected)
        self.timeline.pack(fill="x", padx=16, pady=(0, 16))

        self.frame_info_label = ctk.CTkLabel(self.results_view, text="", font=(theme.FONT_FAMILY, 12),
                                              text_color=theme.TEXT_SECONDARY)
        self.frame_info_label.pack(anchor="w", pady=(0, 8))

        ctk.CTkLabel(self.results_view, text="Detected Faces", font=(theme.FONT_FAMILY, 14, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        self.face_gallery = FaceGallery(self.results_view)
        self.face_gallery.pack(fill="x", pady=(0, 16))

        ctk.CTkButton(self.results_view, text="Analyze Another Video", command=self.reset,
                      fg_color=theme.BG_CARD_HOVER, hover_color=theme.BORDER).pack(anchor="w")

    def run_analysis(self, path, on_progress):
        file_hash = self.app.pipeline._get_file_hash(path)
        cached = self.app.store.get_cached_result(file_hash)
        if cached:
            return cached
        report = self.app.pipeline.analyze_media(path, is_image=False, update_progress_cb=on_progress)
        self.app.store.cache_result(file_hash, report)
        return report

    def render_result(self, report: dict):
        self.current_report = report
        mode_line = ("Transformer Model Active (ViT_Deepfake_Detection)" if report.get("used_vit_model")
                     else "Local Heuristics & Image Forensics Active")
        self.gauge_card.render(
            filename=report["filename"],
            mode_line=mode_line,
            fake_score=report["global_fake_score"],
            is_fake=report["is_fake"],
            stats=[
                ("Analysis Target", "Video File"),
                ("Processing Latency", f"{report['processing_time_sec']}s"),
                ("Sampled Frames", report["total_frames_analyzed"]),
                ("Total Faces Detected", report["total_faces_detected"]),
            ],
        )
        self.heuristics_panel.render([
            ("Boundary Blurring", report["average_heuristics"]["blur_artifact_score"]),
            ("Spectral Frequency Anomalies", report["average_heuristics"]["frequency_anomaly_score"]),
            ("Lighting & Skin Tone Outliers", report["average_heuristics"]["color_anomaly_score"]),
        ])

        points = []
        for frame in report.get("frames", []):
            score = max((f["fake_score"] for f in frame["faces"]), default=0.0)
            points.append((frame["frame_idx"], score))
        self.timeline.set_points(points, selected_idx=points[0][0] if points else 0)

        if points:
            self._show_frame(points[0][0])

    def _on_frame_selected(self, frame_idx: int):
        self._show_frame(frame_idx)

    def _show_frame(self, frame_idx: int):
        if not self.current_report:
            return
        frame = next((f for f in self.current_report["frames"] if f["frame_idx"] == frame_idx), None)
        if not frame:
            return
        self.frame_info_label.configure(
            text=f"Frame #{frame_idx + 1} @ {frame['timestamp']}s — {len(frame['faces'])} face(s) detected"
        )
        self.face_gallery.render(frame["faces"])
