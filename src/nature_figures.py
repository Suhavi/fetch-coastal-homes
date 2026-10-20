"""Golden-ratio Matplotlib figures for Nature print and presentations.

``print`` uses 12 pt text on a 2x editing canvas. :func:`save_figure`
reduces that canvas to the requested Nature width, making ordinary text 6 pt
and panel letters 8 pt in the exported file. ``ppt`` uses 18 pt text on a
larger canvas (12 in wide for ``double``) without reduction. Use
:func:`save_figure` for print-ready output; direct ``fig.savefig`` will retain
the larger editing canvas.

Nature's figure rules: https://www.nature.com/nature/for-authors/formatting-guide
and https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
The golden ratio is a lab preference, not a Nature requirement.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.text import Text


MM_TO_IN = 1 / 25.4
PHI = (1 + 5 ** 0.5) / 2


NatureWidth = Literal["single", "onehalf", "onehalf_wide", "double"]
FigureMode = Literal["print", "ppt"]

MAX_NATURE_HEIGHT_MM = 170
PRINT_WORK_SCALE = 2
PPT_DOUBLE_WIDTH_IN = 12
PPT_WORK_SCALE = PPT_DOUBLE_WIDTH_IN / (183 * MM_TO_IN)


NATURE_WIDTHS_MM: dict[NatureWidth, float] = {
    "single": 89,
    "onehalf": 120,
    "onehalf_wide": 136,
    "double": 183,
}


@dataclass(frozen=True)
class NatureStyle:
    """Editable figure settings; ``print`` text is halved on export."""

    font_family: str = "Helvetica"
    font_size: float = 12
    axis_label_size: float = 12
    title_size: float = 12
    tick_label_size: float = 10
    legend_font_size: float = 12
    annotation_font_size: float = 10
    panel_label_size: float = 16
    axes_linewidth: float = 0.5
    line_width: float = 1.0
    marker_size: float = 3.0

    @classmethod
    def for_mode(cls, mode: FigureMode) -> NatureStyle:
        """Use consistent text sizes for the selected output context."""
        if mode == "print":
            return cls()
        if mode == "ppt":
            return cls(
                font_size=18,
                axis_label_size=22,
                title_size=24,
                tick_label_size=18,
                legend_font_size=18,
                annotation_font_size=14,
                panel_label_size=18,
            )
        raise ValueError("mode must be 'print' or 'ppt'.")


def mm_to_in(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm * MM_TO_IN


def nature_figsize(
    width: NatureWidth | float = "single",
    *,
    height: float | None = None,
    height_mm: float | None = None,
    aspect: float | Literal["golden"] = "golden",
) -> tuple[float, float]:
    """
    Return matplotlib figsize in inches.

    Parameters
    ----------
    width:
        Either one of {"single", "onehalf", "onehalf_wide", "double"}
        or a numeric width in millimeters.

    height:
        Optional explicit height in inches.

    height_mm:
        Optional explicit height in millimeters.

    aspect:
        If "golden", height = width / phi.
        If numeric, height = width * aspect, where width/height are in inches.

    Examples
    --------
    >>> nature_figsize("single")
    (3.5039..., 2.1656...)

    >>> nature_figsize("double")
    (7.2047..., 4.4528...)

    >>> nature_figsize(100, aspect=0.75)
    (3.9370..., 2.9527...)
    """
    if isinstance(width, str):
        width_mm = NATURE_WIDTHS_MM[width]
    else:
        width_mm = float(width)

    width_in = mm_to_in(width_mm)

    if height is not None and height_mm is not None:
        raise ValueError("Use only one of height or height_mm, not both.")

    if height is not None:
        height_in = float(height)
    elif height_mm is not None:
        height_in = mm_to_in(float(height_mm))
    elif aspect == "golden":
        height_in = width_in / PHI
    else:
        height_in = width_in * float(aspect)

    if not 0 < width_mm <= NATURE_WIDTHS_MM["double"]:
        raise ValueError("Nature figure width must be greater than 0 and at most 183 mm.")
    if not 0 < height_in <= mm_to_in(MAX_NATURE_HEIGHT_MM):
        raise ValueError("Nature figure height must be greater than 0 and at most 170 mm.")

    return width_in, height_in


def set_nature_style(
    style: NatureStyle | None = None, *, mode: FigureMode = "print"
) -> None:
    """
    Apply Nature-style matplotlib rcParams globally.

    Call this once near the top of a notebook or script.
    """
    if style is None:
        style = NatureStyle.for_mode(mode)
    elif mode not in ("print", "ppt"):
        raise ValueError("mode must be 'print' or 'ppt'.")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [style.font_family, "Arial", "Helvetica"],
            "text.usetex": False,
            "font.size": style.font_size,
            "axes.labelsize": style.axis_label_size,
            "axes.titlesize": style.title_size,
            "figure.titlesize": style.title_size,
            "figure.labelsize": style.font_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_font_size,
            "legend.title_fontsize": style.legend_font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.grid": False,
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            "xtick.major.width": style.axes_linewidth,
            "ytick.major.width": style.axes_linewidth,
            "xtick.minor.width": style.axes_linewidth,
            "ytick.minor.width": style.axes_linewidth,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.prop_cycle": cycler(
                color=["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#000000"]
            ),
            "image.cmap": "viridis",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.edgecolor": "black",
            "legend.facecolor": "white",
            "legend.framealpha": 1.0,
        }
    )


def nature_subplots(
    width: NatureWidth | float = "single",
    *,
    nrows: int = 1,
    ncols: int = 1,
    height: float | None = None,
    height_mm: float | None = None,
    aspect: float | Literal["golden"] = "golden",
    style: NatureStyle | None = None,
    mode: FigureMode = "print",
    **kwargs,
):
    """
    Create matplotlib subplots with Nature-style size and rcParams.

    ``print`` is drawn at twice the requested dimensions so 12 pt editing
    text becomes 6 pt when :func:`save_figure` exports it. ``ppt`` uses a
    proportionally larger canvas (12 in wide for ``double``) and 18 pt text.
    Extra kwargs go to ``plt.subplots``.
    """
    set_nature_style(style, mode=mode)
    figsize = nature_figsize(width, height=height, height_mm=height_mm, aspect=aspect)
    work_scale = PRINT_WORK_SCALE if mode == "print" else PPT_WORK_SCALE
    work_size = tuple(value * work_scale for value in figsize)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=work_size, **kwargs)
    fig._nature_mode = mode
    fig._nature_style = style if style is not None else NatureStyle.for_mode(mode)
    return fig, axes


def add_panel_labels(
    axes,
    labels: Sequence[str] | None = None,
    *,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: float | None = None,
    fontweight: str = "bold",
) -> None:
    """
    Add lowercase bold panel labels to axes.

    Parameters
    ----------
    axes:
        A single axis or an iterable/array of axes.

    labels:
        Defaults to a, b, c, ...

    x, y:
        Label position in axes coordinates.
    """
    try:
        flat_axes = axes.ravel()
    except AttributeError:
        if isinstance(axes, Iterable):
            flat_axes = list(axes)
        else:
            flat_axes = [axes]

    if labels is None:
        labels = [chr(ord("a") + i) for i in range(len(flat_axes))]

    for ax, label in zip(flat_axes, labels):
        label_size = fontsize
        if label_size is None:
            figure_style = getattr(ax.figure, "_nature_style", NatureStyle())
            label_size = figure_style.panel_label_size
        panel_text = ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=label_size,
            fontweight=fontweight,
            va="top",
            ha="left",
        )
        panel_text.set_gid("nature_panel_label")


@contextmanager
def _export_size(fig, mode: FigureMode) -> Iterator[None]:
    """Temporarily reduce the print editing canvas to final output size."""
    if mode != "print":
        yield
        return

    original_size = fig.get_size_inches().copy()
    # Drawing first creates tick labels, including labels from secondary axes.
    fig.canvas.draw()
    text_sizes = [(item, item.get_fontsize()) for item in fig.findobj(match=Text)]
    for item, size in text_sizes:
        if not item.get_visible() or not item.get_text().strip():
            continue
        final_size = size / PRINT_WORK_SCALE
        if item.get_gid() == "nature_panel_label":
            if final_size != 8:
                raise ValueError("Nature panel labels must be 8 pt at final print size.")
        elif not 5 <= final_size <= 7:
            raise ValueError(
                f"Nature figure text must be 5–7 pt at final print size; "
                f"{item.get_text()!r} would be {final_size:g} pt."
            )
    try:
        for item, size in text_sizes:
            item.set_fontsize(size / PRINT_WORK_SCALE)
        fig.set_size_inches(original_size / PRINT_WORK_SCALE, forward=False)
        yield
    finally:
        fig.set_size_inches(original_size, forward=False)
        for item, size in text_sizes:
            item.set_fontsize(size)


def clean_axes(
    ax,
    *,
    top: bool = True,
    right: bool = True,
    outward: float = 0,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    bounds_are_limits: bool = True,
    color: str = "black",
    spine_linewidth: float = 1.0,
    tick_length: float = 5,
    tick_linewidth: float = 1.0,
) -> None:
    """
    Style axes with dark spines, boxed by default.

    Set top or right to ``False`` to hide that side of the box.
    x_bounds and y_bounds trim the visible spine ends to the first/last
    labeled ticks. By default they also set the matching axis limits so the
    spines stay connected at the lower-left corner.
    """
    visible_spines = {
        "left": True,
        "bottom": True,
        "top": top,
        "right": right,
    }
    for side, visible in visible_spines.items():
        ax.spines[side].set_visible(visible)
        if visible:
            ax.spines[side].set_color(color)
            ax.spines[side].set_linewidth(spine_linewidth)
            ax.spines[side].set_position(("outward", outward))

    if x_bounds is not None:
        if bounds_are_limits:
            ax.set_xlim(*x_bounds)
        ax.spines["bottom"].set_bounds(*x_bounds)
    if y_bounds is not None:
        if bounds_are_limits:
            ax.set_ylim(*y_bounds)
        ax.spines["left"].set_bounds(*y_bounds)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=tick_length,
        width=tick_linewidth,
        colors=color,
    )
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.title.set_color(color)


def save_figure(
    fig,
    path: str | Path,
    *,
    formats: Sequence[str] = ("pdf",),
    dpi: int = 450,
    bbox_inches: str | None = None,
    pad_inches: float = 0.02,
    mode: FigureMode | None = None,
) -> list[Path]:
    """
    Save one figure to one or more formats.

    Examples
    --------
    >>> save_figure(fig, "figures/slr_projection", formats=("pdf", "png"))
    saves figures/slr_projection.pdf and figures/slr_projection.png

    The default keeps the exact golden-ratio canvas. Pass ``bbox_inches="tight"``
    explicitly if cropping matters more than the specified image dimensions.
    ``pdf`` is preferred for final Nature line art; other formats remain
    available for presentations, previews, or other submission contexts.
    """
    if mode is None:
        mode = getattr(fig, "_nature_mode", "ppt")
    if mode not in ("print", "ppt"):
        raise ValueError("mode must be 'print' or 'ppt'.")
    if mode == "print" and dpi < 450:
        raise ValueError("Nature print raster content must be exported at 450 dpi or higher.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    with _export_size(fig, mode):
        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            output = path.with_suffix(f".{fmt}")
            fig.savefig(
                output,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
            )
            saved_paths.append(output)

    return saved_paths


def nature_lineplot_example(mode: FigureMode = "print"):
    """
    Minimal example. Delete or ignore in production.
    """
    fig, ax = nature_subplots("single", mode=mode)
    ax.plot([2026, 2036, 2046], [0, 0.26, 0.58])
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea-level rise (ft)")
    clean_axes(ax)
    return fig, ax

if __name__ == "__main__":
    nature_lineplot_example()
    plt.show()
