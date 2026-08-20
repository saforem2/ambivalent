"""
Tests for ambivalent.core.FONTS_DIR location resolution.

FONTS_DIR must not depend on the current working directory (an import-time
coupling that breaks when the launch dir differs or is read-only). It should
honor an explicit override and otherwise fall back to a stable per-user cache.
"""
from __future__ import annotations

import importlib
from pathlib import Path


def _reload_core(monkeypatch, env: dict, cwd: Path):
    for key in ("AMBIVALENT_FONTS_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    monkeypatch.chdir(cwd)
    import ambivalent.core as core
    return importlib.reload(core)


def test_fonts_dir_honors_explicit_env(tmp_path, monkeypatch):
    target = tmp_path / "my-fonts"
    core = _reload_core(
        monkeypatch, {"AMBIVALENT_FONTS_DIR": str(target)}, cwd=tmp_path
    )
    assert core.FONTS_DIR == target


def test_fonts_dir_independent_of_cwd(tmp_path, monkeypatch):
    """Same env, two different cwds -> same FONTS_DIR."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    core_a = _reload_core(monkeypatch, {}, cwd=a)
    dir_a = core_a.FONTS_DIR
    core_b = _reload_core(monkeypatch, {}, cwd=b)
    dir_b = core_b.FONTS_DIR
    assert dir_a == dir_b
    # And it must not sit under either cwd.
    assert a not in dir_a.parents and a != dir_a
    assert b not in dir_b.parents and b != dir_b


def test_fonts_dir_uses_xdg_cache_home(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    core = _reload_core(monkeypatch, {"XDG_CACHE_HOME": str(xdg)}, cwd=tmp_path)
    assert xdg in core.FONTS_DIR.parents


def test_fonts_dir_not_created_at_import(tmp_path, monkeypatch):
    target = tmp_path / "lazy-fonts"
    core = _reload_core(
        monkeypatch, {"AMBIVALENT_FONTS_DIR": str(target)}, cwd=tmp_path
    )
    assert core.FONTS_DIR == target
    assert not target.exists(), "FONTS_DIR should not be created at import time"
