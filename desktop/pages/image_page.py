import customtkinter as ctk

from desktop import theme
from desktop.widgets import HeuristicsPanel, FaceGallery
from desktop.pages.base import BaseAnalysisPage


class ImagePage(BaseAnalysisPage):
    def __init__(self, master, app):
        self.app = app
        super().__init__(
            master,
            page_title="Image Deepfake Analysis",
            page_subtitle="Scan a photo for face-swap artifacts, AI-generation traces, and pixel-level forensic anomalies.",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )

        self.heuristics_panel = HeuristicsPanel(self.results_view, "Pixel-Level Visual Artifact Markers")
        self.heuristics_panel.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(self.results_view, text="Detected Faces", font=(theme.FONT_FAMILY, 14, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        self.face_gallery = FaceGallery(self.results_view)
        self.face_gallery.pack(fill="x", pady=(0, 16))

        ctk.CTkButton(self.results_view, text="Analyze Another Image", command=self.reset,
                      fg_color=theme.BG_CARD_HOVER, hover_color=theme.BORDER).pack(anchor="w")

    def run_analysis(self, path, on_progress):
        file_hash = self.app.pipeline._get_file_hash(path)
        cached = self.app.store.get_cached_result(file_hash)
        if cached:
            return cached
        report = self.app.pipeline.analyze_media(path, is_image=True, update_progress_cb=on_progress)
        self.app.store.cache_result(file_hash, report)
        return report

    def render_result(self, report: dict):
        faces = report["frames"][0]["faces"] if report.get("frames") else []
        mode_line = ("Transformer Model Active (ViT_Deepfake_Detection)" if report.get("used_vit_model")
                     else "Local Heuristics & Image Forensics Active")
        self.gauge_card.render(
            filename=report["filename"],
            mode_line=mode_line,
            fake_score=report["global_fake_score"],
            is_fake=report["is_fake"],
            stats=[
                ("Analysis Target", "Image File"),
                ("Processing Latency", f"{report['processing_time_sec']}s"),
                ("Faces Detected", report["total_faces_detected"]),
                ("Confidence", f"{round(report['confidence'] * 100)}%"),
            ],
        )
        self.heuristics_panel.render([
            ("Boundary Blurring", report["average_heuristics"]["blur_artifact_score"]),
            ("Spectral Frequency Anomalies", report["average_heuristics"]["frequency_anomaly_score"]),
            ("Lighting & Skin Tone Outliers", report["average_heuristics"]["color_anomaly_score"]),
        ])
        self.face_gallery.render(faces)
