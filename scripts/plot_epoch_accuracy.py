"""Plot fold-averaged VQC accuracy by epoch from epoch_history.csv."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 1300
LEFT = 250
RIGHT = 1680
TOP = 180
BOTTOM = 1030


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="4-Layer VQC Accuracy by Epoch",
    )
    return parser


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    candidates = [
        Path("C:/Windows/Fonts") / font_name,
        Path(ImageFont.__file__).parent / "fonts" / "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def vertical_text(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    text_font: ImageFont.ImageFont,
) -> None:
    bounds = ImageDraw.Draw(image).textbbox((0, 0), text, font=text_font)
    label = Image.new(
        "RGBA",
        (bounds[2] - bounds[0] + 20, bounds[3] - bounds[1] + 20),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(label).text(
        (10, 10 - bounds[1]),
        text,
        font=text_font,
        fill="#1F1F1F",
    )
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(
        label,
        (center[0] - label.width // 2, center[1] - label.height // 2),
        label,
    )


def aggregate(history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "epoch",
        "fold",
        "training_accuracy",
        "validation_accuracy",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")

    grouped = history.groupby("epoch", sort=True)
    result = grouped.agg(
        folds=("fold", "nunique"),
        training_accuracy_mean=("training_accuracy", "mean"),
        training_accuracy_std=("training_accuracy", "std"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
    ).reset_index()

    if result["folds"].min() < 2:
        raise ValueError("At least two folds are required for a standard-deviation band.")
    return result


def plot(summary: pd.DataFrame, title: str, destination: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (WIDTH / 2, 85),
        title,
        font=font(52, bold=True),
        fill="#1F1F1F",
        anchor="mm",
    )

    epochs = summary["epoch"].astype(float).tolist()
    all_lower: list[float] = []
    all_upper: list[float] = []
    for prefix in ("training", "validation"):
        mean = summary[f"{prefix}_accuracy_mean"] * 100
        deviation = summary[f"{prefix}_accuracy_std"].fillna(0) * 100
        all_lower.extend((mean - deviation).tolist())
        all_upper.extend((mean + deviation).tolist())

    y_min = max(0, math.floor((min(all_lower) - 3) / 5) * 5)
    y_max = min(100, math.ceil((max(all_upper) + 3) / 5) * 5)
    if y_max - y_min < 20:
        y_min = max(0, y_min - 5)
        y_max = min(100, y_max + 5)

    x_min, x_max = min(epochs), max(epochs)

    def x_position(value: float) -> float:
        return LEFT + (value - x_min) / (x_max - x_min) * (RIGHT - LEFT)

    def y_position(value: float) -> float:
        return BOTTOM - (value - y_min) / (y_max - y_min) * (BOTTOM - TOP)

    tick_step = 5 if y_max - y_min <= 40 else 10
    for tick in range(int(y_min), int(y_max) + 1, tick_step):
        y = y_position(float(tick))
        draw.line((LEFT, y, RIGHT, y), fill="#D9D9D9", width=2)
        draw.text(
            (LEFT - 25, y),
            str(tick),
            font=font(29),
            fill="#404040",
            anchor="rm",
        )

    x_ticks = sorted(set([1, *range(10, int(x_max) + 1, 10), int(x_max)]))
    for tick in x_ticks:
        if tick < x_min or tick > x_max:
            continue
        x = x_position(float(tick))
        draw.line((x, BOTTOM, x, BOTTOM + 10), fill="#1F1F1F", width=3)
        draw.text(
            (x, BOTTOM + 28),
            str(tick),
            font=font(29),
            fill="#404040",
            anchor="ma",
        )

    draw.line((LEFT, TOP, LEFT, BOTTOM), fill="#1F1F1F", width=4)
    draw.line((LEFT, BOTTOM, RIGHT, BOTTOM), fill="#1F1F1F", width=4)

    series = [
        ("training", "Training accuracy", "#4472C4", (68, 114, 196, 48)),
        ("validation", "Validation accuracy", "#ED7D31", (237, 125, 49, 48)),
    ]
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for prefix, _label, color, band_color in series:
        mean = (summary[f"{prefix}_accuracy_mean"] * 100).tolist()
        deviation = (
            summary[f"{prefix}_accuracy_std"].fillna(0) * 100
        ).tolist()
        upper = [
            (x_position(epoch), y_position(value + spread))
            for epoch, value, spread in zip(epochs, mean, deviation, strict=True)
        ]
        lower = [
            (x_position(epoch), y_position(value - spread))
            for epoch, value, spread in zip(epochs, mean, deviation, strict=True)
        ]
        overlay_draw.polygon(upper + list(reversed(lower)), fill=band_color)
        points = [
            (x_position(epoch), y_position(value))
            for epoch, value in zip(epochs, mean, strict=True)
        ]
        overlay_draw.line(points, fill=color, width=7, joint="curve")

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    vertical_text(
        image,
        (72, int((TOP + BOTTOM) / 2)),
        "Accuracy (%)",
        font(34),
    )
    draw.text(
        (WIDTH / 2, BOTTOM + 90),
        "Epoch",
        font=font(34),
        fill="#1F1F1F",
        anchor="mm",
    )

    legend_y = 1165
    for x, (_prefix, label, color, _band) in zip(
        (590, 1110),
        series,
        strict=True,
    ):
        draw.line((x - 90, legend_y, x - 20, legend_y), fill=color, width=8)
        draw.text(
            (x, legend_y),
            label,
            font=font(30),
            fill="#1F1F1F",
            anchor="lm",
        )

    draw.text(
        (WIDTH / 2, 1230),
        "Lines show the 5-fold mean; shaded bands show +/- 1 SD",
        font=font(27),
        fill="#595959",
        anchor="mm",
    )
    image.save(destination.with_suffix(".png"), dpi=(300, 300))
    image.save(destination.with_suffix(".pdf"), resolution=300)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(args.history)
    summary = aggregate(history)
    summary.to_csv(args.output_dir / "epoch_accuracy_summary.csv", index=False)
    plot(summary, args.title, args.output_dir / "epoch_accuracy_4layer")


if __name__ == "__main__":
    main()
