"""Shared path inputs for dashboard pages."""

import streamlit as st


def path_input(
    label: str,
    default: str,
    *,
    key: str,
    help: str | None = None,
) -> str:
    """Text input for a repository-relative path."""
    return st.text_input(
        label,
        value=default,
        help=help or "Path relative to the repository root.",
        key=key,
    )
