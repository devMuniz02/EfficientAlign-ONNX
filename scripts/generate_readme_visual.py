from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with `pip install pillow`.") from exc


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUT = ASSETS / "pipeline-overview.png"


def load_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded_box(draw, box, radius, fill, outline, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, box, text, font, fill):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6, align="center")


def arrow(draw, start, end, fill, width=7):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    size = 14
    draw.polygon(
        [(ex, ey), (ex - size, ey - size / 2), (ex - size, ey + size / 2)],
        fill=fill,
    )


def main():
    ASSETS.mkdir(exist_ok=True)

    canvas = Image.new("RGB", (1600, 900), "#0b1220")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(44, bold=True)
    subtitle_font = load_font(24)
    block_title_font = load_font(26, bold=True)
    block_text_font = load_font(18)
    footer_font = load_font(20, bold=True)

    draw.text((70, 52), "EfficientAlign-ONNX", font=title_font, fill="#f8fafc")
    draw.text(
        (70, 112),
        "End-to-end workflow for fine-tuning, alignment, ONNX export, and efficient deployment.",
        font=subtitle_font,
        fill="#cbd5e1",
    )

    blocks = [
        ("Base Model", "Gemma / LLM\nstarting checkpoint", "#1e293b", "#60a5fa"),
        ("QLoRA Fine-Tuning", "low-resource training\nadapter-based updates", "#132238", "#38bdf8"),
        ("DPO / RLHF Alignment", "preference tuning\naligned outputs", "#182536", "#a78bfa"),
        ("ONNX Export", "portable graph\nruntime-friendly artifact", "#192b24", "#34d399"),
        ("Consumer Deployment", "optimized inference\nlocal hardware ready", "#2a2317", "#f59e0b"),
    ]

    top = 290
    left = 55
    block_w = 270
    block_h = 240
    gap = 40

    for idx, (title, body, fill, outline) in enumerate(blocks):
        x0 = left + idx * (block_w + gap)
        x1 = x0 + block_w
        box = (x0, top, x1, top + block_h)
        rounded_box(draw, box, 26, fill, outline)
        center_text(draw, (x0 + 20, top + 24, x1 - 20, top + 110), title, block_title_font, "#f8fafc")
        center_text(draw, (x0 + 24, top + 110, x1 - 24, top + block_h - 30), body, block_text_font, "#dbeafe")
        if idx < len(blocks) - 1:
            y = top + block_h / 2
            arrow(draw, (x1 + 8, y), (x1 + gap - 8, y), "#64748b")

    footer_box = (70, 700, 1530, 820)
    rounded_box(draw, footer_box, 22, "#111827", "#334155", width=2)
    footer = "Training -> Alignment -> Export -> Deployment | Technical, reproducible, consumer-hardware oriented"
    center_text(draw, footer_box, footer, footer_font, "#e2e8f0")

    canvas.save(OUTPUT)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
