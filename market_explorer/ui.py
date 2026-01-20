from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "neat-logo.svg"


def add_corner_logo(
    *,
    width_px: int = 180,
    offset_px: int = 16,
    alt_text: str = "Neat",
) -> None:
    if not LOGO_PATH.exists():
        return

    encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    mobile_width = max(120, width_px - 40)

    st.markdown(
        f"""
        <style>
        .neat-logo {{
            position: fixed;
            top: {offset_px}px;
            left: {offset_px}px;
            width: {width_px}px;
            z-index: 9999;
            opacity: 0.95;
        }}

        @media (max-width: 768px) {{
            .neat-logo {{
                width: {mobile_width}px;
                top: 8px;
                left: 8px;
            }}
        }}
        </style>

        <img class="neat-logo"
             src="data:image/svg+xml;base64,{encoded_logo}"
             alt="{alt_text}" />
        """,
        unsafe_allow_html=True,
    )
