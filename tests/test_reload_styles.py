"""
Tests for ambivalent.reload_styles resilience to read-only config dirs.

The bug: ambivalent copies its style files into matplotlib's config dir at
import time and hard-fails with a PermissionError wherever that dir is
read-only (e.g. shared HPC installs).
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import ambivalent


@pytest.fixture(autouse=True)
def _isolate_matplotlib_style_state():
    """Snapshot/restore matplotlib's process-global style state.

    These tests mutate ``plt.style.library``, ``plt.style.core.available``
    and ``plt.rcParams``, all of which are global. Restoring them keeps the
    suite order-independent as it grows.
    """
    saved_library = dict(plt.style.library)
    saved_available = list(plt.style.core.available)
    saved_rcparams = plt.rcParams.copy()
    try:
        yield
    finally:
        plt.style.library.clear()
        plt.style.library.update(saved_library)
        plt.style.core.available[:] = saved_available
        plt.rcParams.update(saved_rcparams)


def _make_readonly(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x------ : no write


def test_reload_styles_does_not_raise_on_readonly_outdir(tmp_path):
    """A read-only target dir must degrade gracefully, not raise."""
    readonly = tmp_path / "stylelib"
    _make_readonly(readonly)
    try:
        # Should NOT raise PermissionError.
        ambivalent.reload_styles(outdir=readonly)
    finally:
        readonly.chmod(stat.S_IRWXU)  # restore so tmp cleanup works


def test_reload_styles_registers_style_by_name_when_copy_fails(tmp_path):
    """Even when the disk copy fails, the style is usable by name in-session."""
    readonly = tmp_path / "stylelib"
    _make_readonly(readonly)
    try:
        ambivalent.reload_styles(outdir=readonly)
        assert "ambivalent" in plt.style.library
        # Registered by name is not enough -- it must actually apply.
        before = plt.rcParams.copy()
        plt.style.use("ambivalent")
        assert plt.rcParams != before, "style.use('ambivalent') changed nothing"
    finally:
        readonly.chmod(stat.S_IRWXU)


def test_reload_styles_copies_to_writable_outdir(tmp_path):
    """On a writable dir the file is still copied (existing behavior kept)."""
    outdir = tmp_path / "stylelib"
    ambivalent.reload_styles(outdir=outdir)
    copied = list(outdir.glob("ambivalent*"))
    assert copied, f"expected an ambivalent style file copied into {outdir}"
    assert "ambivalent" in plt.style.library


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file perms")
def test_readonly_check_actually_blocks_writes(tmp_path):
    """Guard: confirm our read-only fixture really prevents writes."""
    readonly = tmp_path / "ro"
    _make_readonly(readonly)
    try:
        with pytest.raises((PermissionError, OSError)):
            (readonly / "probe").write_text("x")
    finally:
        readonly.chmod(stat.S_IRWXU)
