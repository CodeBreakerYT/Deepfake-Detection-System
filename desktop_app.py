import os
import sys


def _bootstrap_bundled_model_cache():
    """
    When running as a packaged exe, point huggingface_hub at the model cache
    bundled next to the executable and force offline mode, so the app never
    needs internet access or a pre-existing user cache on the target machine.
    Has no effect when running from source (`python desktop_app.py`).
    """
    if not getattr(sys, "frozen", False):
        return
    # sys._MEIPASS is the PyInstaller bundle's data directory (the "_internal"
    # folder in onedir builds) - this is where our `datas` entries actually land,
    # not next to the exe itself.
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    bundled_cache = os.path.join(base_dir, "hf_cache")
    if os.path.isdir(bundled_cache):
        os.environ["HF_HOME"] = bundled_cache
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_bootstrap_bundled_model_cache()

import threading
import customtkinter as ctk

from desktop import theme
from desktop.animations import fade_window
from desktop.pages.start_page import StartPage
from desktop.pages.about_page import AboutScreen
from desktop.main_shell import MainShell


class DeepShieldApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Deepfake Detection System")
        self.geometry("1300x860")
        self.minsize(1000, 700)
        self.configure(fg_color=theme.BG_MAIN)
        self.attributes("-alpha", 0.0)

        self.pipeline = None
        self.voice_pipeline = None
        self.store = None
        self._services_ready = False

        self.container = ctk.CTkFrame(self, fg_color=theme.BG_MAIN)
        self.container.pack(fill="both", expand=True)

        self.start_page = StartPage(self.container, on_start=self.enter_main, on_quit=self.quit_app,
                                     on_about=self.show_about)
        self.start_page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.about_screen = AboutScreen(self.container, on_back=self.show_start)
        self.main_shell = MainShell(self.container, app=self, on_quit=self.quit_app, on_home=self.show_start)

        self.protocol("WM_DELETE_WINDOW", self.quit_app)

        threading.Thread(target=self._init_services, daemon=True).start()
        self.after(80, lambda: fade_window(self, 0.0, 1.0, steps=16, delay=14))

    # ---- service bootstrap ----

    def _init_services(self):
        from scripts.deepfake_classifier import DeepfakeClassifier
        from scripts.pipeline import DetectionPipeline
        from scripts.voice_classifier import VoiceClassifier
        from scripts.voice_pipeline import VoiceDetectionPipeline
        from scripts.local_store import LocalStore

        classifier = DeepfakeClassifier()
        self.pipeline = DetectionPipeline(classifier)
        voice_classifier = VoiceClassifier()
        self.voice_pipeline = VoiceDetectionPipeline(voice_classifier)
        self.store = LocalStore()

        self.after(0, self._on_services_ready)

    def _on_services_ready(self):
        self._services_ready = True
        self.main_shell.build_pages()
        self.start_page.set_ready()

    # ---- screen navigation ----

    def enter_main(self):
        if not self._services_ready:
            return

        def _swap():
            self.start_page.place_forget()
            self.about_screen.place_forget()
            self.main_shell.place(relx=0, rely=0, relwidth=1, relheight=1)
            fade_window(self, 0.0, 1.0, steps=10, delay=10)

        fade_window(self, 1.0, 0.0, steps=10, delay=10, on_done=_swap)

    def show_start(self):
        def _swap():
            self.main_shell.place_forget()
            self.about_screen.place_forget()
            self.start_page.place(relx=0, rely=0, relwidth=1, relheight=1)
            fade_window(self, 0.0, 1.0, steps=10, delay=10)

        fade_window(self, 1.0, 0.0, steps=10, delay=10, on_done=_swap)

    def show_about(self):
        def _swap():
            self.start_page.place_forget()
            self.about_screen.place(relx=0, rely=0, relwidth=1, relheight=1)
            fade_window(self, 0.0, 1.0, steps=10, delay=10)

        fade_window(self, 1.0, 0.0, steps=10, delay=10, on_done=_swap)

    def quit_app(self):
        fade_window(self, 1.0, 0.0, steps=10, delay=10, on_done=self.destroy)


if __name__ == "__main__":
    app = DeepShieldApp()
    app.mainloop()
