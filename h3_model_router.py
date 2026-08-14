# -*- coding: utf-8 -*-
"""Lazy FL2VA/Ref2VA routing for optimized H3 workflows."""


class H3ModelRoute:
    """Select only the model chains required by the render mode."""

    MODES = ("FL2VA only", "Ref2VA only", "Mixed per segment")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {
                    "default": "FL2VA only",
                    "tooltip": "FL2VA only loads only FL2VA. Ref2VA only loads "
                               "only Ref2VA and routes it as the sampler's "
                               "primary model. Mixed loads both and lets the "
                               "sampler select Ref2VA per routed references.",
                }),
            },
            "optional": {
                "fl2va_model": ("MODEL", {"lazy": True}),
                "ref2va_model": ("MODEL", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model", "ref2va_model")
    FUNCTION = "route"
    CATEGORY = "loaders/minimax"

    def check_lazy_status(
        self, mode, fl2va_model=None, ref2va_model=None
    ):
        required = []
        if mode in ("FL2VA only", "Mixed per segment"):
            if fl2va_model is None:
                required.append("fl2va_model")
        if mode in ("Ref2VA only", "Mixed per segment"):
            if ref2va_model is None:
                required.append("ref2va_model")
        return required

    def route(self, mode, fl2va_model=None, ref2va_model=None):
        if mode == "FL2VA only":
            if fl2va_model is None:
                raise ValueError("H3 Model Route: FL2VA input is not connected")
            return (fl2va_model, None)
        if mode == "Ref2VA only":
            if ref2va_model is None:
                raise ValueError("H3 Model Route: Ref2VA input is not connected")
            return (ref2va_model, None)
        if fl2va_model is None or ref2va_model is None:
            raise ValueError(
                "H3 Model Route: Mixed mode requires both model inputs"
            )
        return (fl2va_model, ref2va_model)


NODE_CLASS_MAPPINGS = {"H3ModelRoute": H3ModelRoute}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ModelRoute": "H3 Model Route (lazy FL2VA / Ref2VA)"
}
