PRIMARY = "#8b5cf6"
SECONDARY = "#06b6d4"
SUCCESS = "#10b981"
DANGER = "#f43f5e"
WARNING = "#f59e0b"

BG_MAIN = "#060913"
BG_CARD = "#0d1426"
BG_CARD_HOVER = "#141e3a"
BORDER = "#23293f"

TEXT_PRIMARY = "#f3f4f6"
TEXT_SECONDARY = "#9ca3af"
TEXT_MUTED = "#6b7280"

FONT_FAMILY = "Segoe UI"

def heuristic_color(score: float) -> str:
    if score > 0.6:
        return DANGER
    if score > 0.3:
        return WARNING
    return SUCCESS
