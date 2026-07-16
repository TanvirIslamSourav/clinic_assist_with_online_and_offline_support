"""Reusable static artifact display helpers."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd
import streamlit as st


def display_optional_image(
    path: Path,
    caption: str | None = None,
    use_container_width: bool = False,
    width: int | None = 560,
) -> None:
    """Display image if present, otherwise show informative warning."""
    if path.exists() and path.is_file():
        if use_container_width:
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.image(str(path), caption=caption, width=width)
    else:
        st.warning(f"Missing figure: {path}")


def display_image_group(
    paths: list[Path],
    title: str | None = None,
    columns: int = 2,
    image_width: int | None = 560,
    use_container_width: bool = True,
) -> None:
    """Display a group of images using a responsive grid."""
    if title:
        st.subheader(title)

    if not paths:
        st.info("No figures configured.")
        return

    column_count = max(1, min(columns, len(paths)))
    grid_columns = st.columns(column_count)
    for index, path in enumerate(paths):
        with grid_columns[index % column_count]:
            display_optional_image(
                path=path,
                caption=path.name,
                use_container_width=use_container_width,
                width=image_width,
            )


def display_optional_csv(path: Path, title: str) -> None:
    """Display CSV table if file exists."""
    st.subheader(title)
    if not path.exists() or not path.is_file():
        st.warning(f"Missing table: {path}")
        return
    dataframe = pd.read_csv(path)
    st.dataframe(dataframe, use_container_width=True)


def display_hero_image(path: Path, caption: str | None = None) -> None:
    """Display an image as a large full-width responsive hero figure."""
    if not path.exists() or not path.is_file():
        st.warning(f"Missing figure: {path}")
        return

    suffix = path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")

    encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
    safe_caption = html.escape(caption or path.name)
    st.markdown(
        f"""
        <div class="hero-figure-wrapper">
            <img class="hero-figure-image" src="data:{mime_type};base64,{encoded_image}" alt="{safe_caption}" />
            <div class="hero-figure-caption">{safe_caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
