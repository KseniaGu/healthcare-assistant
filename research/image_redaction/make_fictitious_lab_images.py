"""Create tiny invented laboratory-report images for local redactor demonstrations."""

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1100, 850), "white")
    draw = ImageDraw.Draw(image)
    regular = font(29)
    header = font(38)
    lines = [
        ("FICTITIOUS LABORATORY REPORT", 45, header),
        ("Patient: Ada Inventova", 125, regular),
        ("DOB: 01.02.1980    Order: AB1234", 175, regular),
        ("Email: ada@example.invalid", 225, regular),
        ("Phone: +7 999 000 00 11", 275, regular),
        ("Collection date: 03.04.2024", 325, regular),
        ("Test                    Result           Reference", 405, regular),
        ("Glucose                 5.2 mmol/L       3.9-5.8", 465, regular),
        ("Hemoglobin             139 g/L           120-160", 515, regular),
        ("Doctor: Iva Fabricova", 660, regular),
    ]
    for text, y, text_font in lines:
        draw.text((50, y), text, fill="black", font=text_font)
    qr = cv2.QRCodeEncoder_create().encode("https://example.invalid/report/FAKE-001")
    qr_image = Image.fromarray(qr).convert("RGB").resize((220, 220), Image.Resampling.NEAREST)
    image.paste(qr_image, (820, 60))
    image.save(args.output / "invented_report.png")
    blank = Image.new("RGB", (1100, 850), "white")
    blank_draw = ImageDraw.Draw(blank)
    blank_draw.text((50, 45), "FICTITIOUS BLANK LAB FORM", fill="black", font=header)
    blank_draw.text((50, 130), "Patient: ____________________", fill="black", font=regular)
    blank_draw.text((50, 220), "Test                 Result                 Units", fill="black", font=regular)
    blank.save(args.output / "invented_blank_form.png")
    print("Created two fictitious single-frame images")


if __name__ == "__main__":
    main()
