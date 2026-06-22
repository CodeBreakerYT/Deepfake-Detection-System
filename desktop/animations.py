"""Small, dependency-free animation helpers built on Tk's .after() loop."""


def fade_window(root, start: float, end: float, steps: int = 12, delay: int = 12, on_done=None):
    """Animates the whole Toplevel/CTk window's alpha (Tk supports per-window alpha only)."""
    step_val = (end - start) / steps

    def _step(i, val):
        try:
            root.attributes("-alpha", max(0.0, min(1.0, val)))
        except Exception:
            return
        if i < steps:
            root.after(delay, lambda: _step(i + 1, val + step_val))
        elif on_done:
            on_done()

    _step(0, start)


def slide_in(widget, container, direction: str = "right", steps: int = 14, delay: int = 8):
    """
    Slides a widget into place using the place() geometry manager.
    direction is where the widget animates FROM ("right" or "left").
    """
    start_x = 0.35 if direction == "right" else -0.35
    widget.place(relx=start_x, rely=0, relwidth=1, relheight=1)
    widget.lift()

    def _step(i):
        progress = (i + 1) / steps
        # ease-out cubic
        eased = 1 - (1 - progress) ** 3
        x = start_x * (1 - eased)
        widget.place(relx=x, rely=0, relwidth=1, relheight=1)
        if i + 1 < steps:
            widget.after(delay, lambda: _step(i + 1))
        else:
            widget.place(relx=0, rely=0, relwidth=1, relheight=1)

    _step(0)


def pulse_widget(widget, color_a: str, color_b: str, prop: str = "border_color", interval: int = 700):
    """
    Gently alternates a widget's color property between two values forever.
    Returns a cancel() function to stop the loop (call it before destroying the widget).
    """
    state = {"on": False, "cancelled": False, "job": None}

    def _tick():
        if state["cancelled"]:
            return
        try:
            widget.configure(**{prop: color_b if state["on"] else color_a})
        except Exception:
            return
        state["on"] = not state["on"]
        state["job"] = widget.after(interval, _tick)

    _tick()

    def cancel():
        state["cancelled"] = True
        if state["job"]:
            try:
                widget.after_cancel(state["job"])
            except Exception:
                pass

    return cancel
