"""
src/ambivalent/__init__.py
"""
# ruff: noqa: F401
from __future__ import (
    absolute_import,
    annotations,
    division,
    nested_scopes,
    print_function,
)

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

from ambivalent.core import (
    # FONT_NAMES,
    FONTS_DIR,
    PROJECT_DIR,
    STYLES,
    STYLES_DIR,
    download_googlefont,
    update_matplotlib_fonts,
)
# from ambivalent.utils import (
#     # add_attribution,
#     # add_legend,
#     # set_title_and_suptitle
# )

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

os.environ['PROJECT_DIR'] = os.path.abspath(PROJECT_DIR)

# __all__ = [
#     'FONTS_DIR',
#     'STYLES_DIR',
#     'PROJECT_DIR',
#     'download_googlefont',
#     'update_matplotlib_fonts',
#     'add_legend',
#     # 'add_attribution',
#     'set_title_and_suptitle'
# ]


def _copy_styles_to_configdir(
        outdir: Path,
        verbose: Optional[bool] = False,
) -> None:
    """Copy the packaged ``.mplstyle`` files into ``outdir``.

    May raise ``PermissionError``/``OSError`` if ``outdir`` (typically
    matplotlib's config ``stylelib``) is not writable; callers handle that.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for src in STYLES.values():
        dst = outdir.joinpath(Path(src).stem)
        if verbose:
            log.debug(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)


def _register_styles_in_memory() -> None:
    """Register ambivalent's styles in matplotlib's in-memory library.

    This makes ``plt.style.use('ambivalent')`` work by name for the current
    process without touching disk. Must run *after* any
    ``plt.style.reload_library()`` call, since reloading rebuilds the library
    from disk and would otherwise discard these entries.
    """
    ambivalent_stylesheets = plt.style.core.read_style_directory(STYLES_DIR)
    plt.style.core.update_nested_dict(
        plt.style.library,
        ambivalent_stylesheets,
    )
    # Update the list of available styles to match the library.
    plt.style.core.available[:] = sorted(plt.style.library.keys())


def reload_styles(
        outdir: Optional[os.PathLike] = None,
        verbose: Optional[bool] = False,
):
    outdir = (
        Path(mpl.get_configdir()).joinpath('stylelib')
        if outdir is None else Path(outdir)
    )
    # Best-effort disk copy so `plt.style.use('ambivalent')` also persists
    # across processes. Never fatal: shared/HPC installs often have a
    # read-only matplotlib config dir, and importing ambivalent must not
    # crash there. On failure we fall back to in-memory registration below.
    try:
        _copy_styles_to_configdir(outdir, verbose=verbose)
    except (PermissionError, OSError) as e:
        if verbose:
            log.debug(
                f"Could not copy styles to {outdir} ({e}); "
                "registering styles in-memory instead."
            )
    else:
        if verbose:
            log.debug(f"Styles persisted to {outdir}.")

    # Reload from disk FIRST (picks up the copy above when it succeeded),
    # then register in-memory LAST so the styles are always available by
    # name this session even when the copy failed. Nothing wipes them after.
    plt.style.reload_library()
    _register_styles_in_memory()


def check_if_font_already_present(font):
    return FONTS_DIR.joinpath(font).exists()


def download_font_with_retry(
        font: str,
        retries: int = 3,
        delay: int = 3,
        verbose: Optional[bool] = False,
) -> None:
    for i in range(retries):
        try:
            if verbose:
                log.debug(f"Now downloading: {font}")
            download_googlefont(font=font)
            return  # return if the download was successful
        except Exception as e:
            if i < retries - 1:  # i is zero indexed
                if verbose:
                    log.debug(
                        f"Attempt {i+1} to download {font} failed with error:"
                        f"{str(e)}. Retrying in {delay} seconds."
                    )
                time.sleep(delay)
            else:
                if verbose:
                    log.debug(
                        f"All attempts to download {font} failed."
                        "Please check your connection and the font name."
                    )
                raise


def update_fonts(font_names: list, verbose: Optional[bool] = False):
    for font in font_names:
        if FONTS_DIR.joinpath(f"{font}.zip").is_file():
            if verbose:
                log.debug(f"{font} already downloaded, continuing!")
            continue
        if not check_if_font_already_present(font):
            download_font_with_retry(font)
    update_matplotlib_fonts()


# update_fonts()
# reload_styles()
