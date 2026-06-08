from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def create_sample(path: Path, intensity: str) -> None:
    size = 512
    image = Image.new("RGB", (size, size), "#03060c")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 70, 442, 442), fill="#101820", outline="#2bb8ff", width=3)

    colors = {
        "low": ["#224466", "#346a88"],
        "medium": ["#215f8f", "#ffcc55", "#ff8844"],
        "high": ["#2bb8ff", "#ffe066", "#ff5b45", "#ffffff"],
    }[intensity]
    points = {"low": 12, "medium": 24, "high": 44}[intensity]
    for i in range(points):
        x = 120 + ((i * 73) % 270)
        y = 115 + ((i * 47) % 280)
        r = 3 + (i % 8)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colors[i % len(colors)])
    image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for level in ("low", "medium", "high"):
        create_sample(root / f"solar_{level}.png", level)

