#!/usr/bin/env python3
"""
Pinout Diagram to KiCad Symbol Converter

Converts a picture (or PDF page) of an IC pinout diagram into a KiCad 6+
.kicad_sym symbol file using Claude Vision API for extraction.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:
    sys.exit("Error: 'anthropic' package not installed. Run: pip install anthropic")

try:
    from PIL import Image
except ImportError:
    sys.exit("Error: 'Pillow' package not installed. Run: pip install Pillow")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_SIDES = ("left", "right", "top", "bottom")
VALID_PIN_TYPES = (
    "input",
    "output",
    "bidirectional",
    "tri_state",
    "passive",
    "free",
    "unspecified",
    "power_in",
    "power_out",
    "open_collector",
    "open_emitter",
    "no_connect",
)

# Map from KiCad electrical type to the .kicad_sym keyword
KICAD_PIN_TYPE_MAP = {
    "input": "input",
    "output": "output",
    "bidirectional": "bidirectional",
    "tri_state": "tri_state",
    "passive": "passive",
    "free": "free",
    "unspecified": "unspecified",
    "power_in": "power_in",
    "power_out": "power_out",
    "open_collector": "open_collector",
    "open_emitter": "open_emitter",
    "no_connect": "no_connect",
}


@dataclass
class Pin:
    number: str
    name: str
    side: str = "left"
    electrical_type: str = "unspecified"
    position: int = 0  # visual order within side (1-indexed, top-to-bottom or left-to-right)

    def __post_init__(self):
        if self.side not in VALID_SIDES:
            self.side = "left"
        if self.electrical_type not in VALID_PIN_TYPES:
            self.electrical_type = "unspecified"


# ---------------------------------------------------------------------------
# Image / PDF loading
# ---------------------------------------------------------------------------


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load an image file and return (base64_data, media_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/png",  # convert BMP to PNG
    }
    if suffix not in media_map:
        sys.exit(f"Unsupported image format: {suffix}")

    if suffix == ".bmp":
        from io import BytesIO

        img = Image.open(path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode(), "image/png"

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()
    return data, media_map[suffix]


def extract_pdf_page_as_image(pdf_path: str, page_num: int) -> tuple[str, str]:
    """Extract a single page from a PDF as a PNG image, return (base64, media_type)."""
    try:
        import fitz
    except ImportError:
        sys.exit("Error: 'PyMuPDF' package not installed. Run: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    if page_num < 1 or page_num > len(doc):
        sys.exit(f"Page {page_num} out of range (PDF has {len(doc)} pages)")

    page = doc[page_num - 1]
    # Render at 2x resolution for better OCR
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    png_data = pix.tobytes("png")
    doc.close()
    return base64.standard_b64encode(png_data).decode(), "image/png"


def get_pdf_page_count(pdf_path: str) -> int:
    try:
        import fitz
    except ImportError:
        sys.exit("Error: 'PyMuPDF' package not installed. Run: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# ---------------------------------------------------------------------------
# Claude Vision API extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are analyzing an image of an IC/component pinout diagram. Extract ALL pins shown in the diagram.

Return a JSON object with this exact structure (no markdown fencing, just raw JSON):
{
  "component_name": "the IC or component name shown in the diagram",
  "pins": [
    {
      "number": "1",
      "name": "VCC",
      "side": "left",
      "type": "power_in",
      "position": 1
    }
  ]
}

Rules:
- "number": The pin number as a string. Use the number shown in the diagram.
- "name": The pin name/label exactly as shown.
- "side": Which side of the IC body the pin is on. One of: "left", "right", "top", "bottom".
  For a standard DIP pinout viewed from the top, pins typically go down the left side
  and back up the right side.
- "type": The electrical type. One of:
  "input", "output", "bidirectional", "tri_state", "passive", "power_in", "power_out",
  "open_collector", "open_emitter", "no_connect", "unspecified".
  Use "power_in" for VCC/VDD/supply pins, "power_in" for GND/VSS pins,
  "input" for obviously input-only pins, "output" for output-only pins,
  "bidirectional" for I/O pins, and "unspecified" if uncertain.
- "position": The visual order of this pin within its side, starting at 1.
  For LEFT and RIGHT sides: 1 = topmost pin, 2 = next pin down, etc. (top to bottom)
  For TOP and BOTTOM sides: 1 = leftmost pin, 2 = next pin right, etc. (left to right)
  This MUST reflect the actual spatial arrangement shown in the diagram image, not the
  pin number order. Pins must appear in the generated symbol at the same relative
  positions they occupy in the diagram.

Important:
- Extract EVERY pin visible in the diagram. Do not skip any.
- The position values are CRITICAL: they must match the visual layout in the image.
  Look carefully at the vertical/horizontal arrangement of each pin on its side.
- If the diagram shows a chip from the top, left side pins typically have low numbers.
- Return ONLY valid JSON. No explanation, no markdown code fences.
"""


def extract_pins_with_claude(
    image_b64: str, media_type: str, api_key: str | None = None
) -> tuple[str, list[Pin]]:
    """Send image to Claude Vision API and extract pin information."""
    client = anthropic.Anthropic(api_key=api_key)

    print("Sending image to Claude Vision API for analysis...")
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )

    raw_text = response.content[0].text.strip()

    # Try to parse JSON - strip markdown fences if present
    json_text = raw_text
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_text = "\n".join(lines)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse Claude response as JSON: {e}")
        print(f"Raw response:\n{raw_text}")
        sys.exit(1)

    component_name = data.get("component_name", "Unknown_IC")
    pins = []
    for p in data.get("pins", []):
        pins.append(
            Pin(
                number=str(p.get("number", "?")),
                name=str(p.get("name", "?")),
                side=p.get("side", "left"),
                electrical_type=p.get("type", "unspecified"),
                position=int(p.get("position", 0)),
            )
        )

    # If positions weren't provided or are all zero, assign them from list order
    for side in VALID_SIDES:
        side_pins = [p for p in pins if p.side == side]
        if side_pins and all(p.position == 0 for p in side_pins):
            for i, p in enumerate(side_pins, 1):
                p.position = i

    return component_name, pins


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------


def print_pin_table(pins: list[Pin]):
    """Print pins in a formatted table."""
    print(f"\n{'Idx':<5} {'Number':<8} {'Name':<20} {'Side':<8} {'Pos':<5} {'Type':<16}")
    print("-" * 65)
    for i, p in enumerate(pins):
        print(
            f"{i:<5} {p.number:<8} {p.name:<20} {p.side:<8} {p.position:<5} {p.electrical_type:<16}"
        )
    print()


def interactive_review(component_name: str, pins: list[Pin]) -> tuple[str, list[Pin]]:
    """Let the user review and edit extracted pins interactively."""
    print(f"\nComponent: {component_name}")
    print(f"Extracted {len(pins)} pins:")
    print_pin_table(pins)

    while True:
        print("Options:")
        print("  [a] Accept and generate symbol")
        print("  [e] Edit a pin")
        print("  [d] Delete a pin")
        print("  [n] Add a new pin")
        print("  [r] Rename component")
        print("  [p] Print pin table again")
        print("  [q] Quit without saving")

        choice = input("\nChoice: ").strip().lower()

        if choice == "a":
            if not pins:
                print("No pins to generate! Add at least one pin first.")
                continue
            return component_name, pins

        elif choice == "e":
            try:
                idx = int(input("Pin index to edit: "))
                if idx < 0 or idx >= len(pins):
                    print(f"Invalid index. Must be 0-{len(pins)-1}")
                    continue
            except ValueError:
                print("Invalid input.")
                continue

            pin = pins[idx]
            print(f"\nEditing pin {idx}: #{pin.number} {pin.name} ({pin.side}, pos={pin.position}, {pin.electrical_type})")
            print("Press Enter to keep current value.\n")

            new_num = input(f"  Number [{pin.number}]: ").strip()
            if new_num:
                pin.number = new_num

            new_name = input(f"  Name [{pin.name}]: ").strip()
            if new_name:
                pin.name = new_name

            new_side = input(f"  Side [{pin.side}] (left/right/top/bottom): ").strip().lower()
            if new_side:
                if new_side in VALID_SIDES:
                    pin.side = new_side
                else:
                    print(f"  Invalid side '{new_side}', keeping '{pin.side}'")

            new_pos = input(f"  Position [{pin.position}] (order within side, 1=top/left): ").strip()
            if new_pos:
                try:
                    pin.position = int(new_pos)
                except ValueError:
                    print(f"  Invalid position '{new_pos}', keeping {pin.position}")

            new_type = input(f"  Type [{pin.electrical_type}]: ").strip().lower()
            if new_type:
                if new_type in VALID_PIN_TYPES:
                    pin.electrical_type = new_type
                else:
                    print(f"  Invalid type '{new_type}', keeping '{pin.electrical_type}'")
                    print(f"  Valid types: {', '.join(VALID_PIN_TYPES)}")

            print("Pin updated.")
            print_pin_table(pins)

        elif choice == "d":
            try:
                idx = int(input("Pin index to delete: "))
                if idx < 0 or idx >= len(pins):
                    print(f"Invalid index. Must be 0-{len(pins)-1}")
                    continue
            except ValueError:
                print("Invalid input.")
                continue
            removed = pins.pop(idx)
            print(f"Deleted pin #{removed.number} ({removed.name})")
            print_pin_table(pins)

        elif choice == "n":
            num = input("  Pin number: ").strip()
            name = input("  Pin name: ").strip()
            side = input("  Side (left/right/top/bottom) [left]: ").strip().lower() or "left"
            pos_str = input("  Position (order within side, 1=top/left) [auto]: ").strip()
            if pos_str:
                try:
                    pos = int(pos_str)
                except ValueError:
                    pos = 0
            else:
                # Auto-assign: next position on that side
                existing = [p.position for p in pins if p.side == side]
                pos = max(existing, default=0) + 1
            etype = input("  Type [unspecified]: ").strip().lower() or "unspecified"
            pins.append(Pin(number=num, name=name, side=side, electrical_type=etype, position=pos))
            print("Pin added.")
            print_pin_table(pins)

        elif choice == "r":
            new_name = input(f"New component name [{component_name}]: ").strip()
            if new_name:
                component_name = new_name
                print(f"Component renamed to: {component_name}")

        elif choice == "p":
            print(f"\nComponent: {component_name}")
            print_pin_table(pins)

        elif choice == "q":
            print("Exiting without saving.")
            sys.exit(0)

        else:
            print("Unknown option.")


# ---------------------------------------------------------------------------
# KiCad symbol generation
# ---------------------------------------------------------------------------

PIN_LENGTH = 2.54  # mm (100 mil)
PIN_SPACING = 2.54  # mm between pins
FONT_SIZE = 1.27  # mm
BODY_MIN_WIDTH = 10.16  # mm (400 mil) minimum body width


def _sanitize_name(name: str) -> str:
    """Make a name safe for KiCad symbol identifiers."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def generate_kicad_symbol(
    component_name: str, pins: list[Pin], output_path: str
):
    """Generate a .kicad_sym file from the component name and pin list."""
    safe_name = _sanitize_name(component_name)

    # Group pins by side and sort by position to preserve diagram layout
    sides: dict[str, list[Pin]] = {"left": [], "right": [], "top": [], "bottom": []}
    for p in pins:
        sides[p.side].append(p)
    for side in sides:
        sides[side].sort(key=lambda p: p.position)

    # Calculate body dimensions
    max_vertical = max(len(sides["left"]), len(sides["right"]), 1)
    max_horizontal = max(len(sides["top"]), len(sides["bottom"]), 1)

    # Body height: enough room for the tallest vertical side
    body_height = max(max_vertical * PIN_SPACING + PIN_SPACING, 5.08)
    # Round up to nearest 2.54
    body_height = _round_to_grid(body_height)

    # Body width: enough room for the widest horizontal side, and for pin names
    # Estimate max pin name length for width
    all_names = [p.name for p in pins if p.side in ("left", "right")]
    max_name_len = max((len(n) for n in all_names), default=3)
    name_width = max(max_name_len * FONT_SIZE * 0.7, BODY_MIN_WIDTH)
    horiz_width = max(max_horizontal * PIN_SPACING + PIN_SPACING, 5.08)
    body_width = max(name_width, horiz_width)
    body_width = _round_to_grid(body_width)

    half_w = body_width / 2
    half_h = body_height / 2

    # Build pin lines
    pin_lines = []

    # Left side pins: point right (angle 0), placed left of body
    for i, p in enumerate(sides["left"]):
        y = half_h - PIN_SPACING - i * PIN_SPACING
        x = -half_w - PIN_LENGTH
        pin_lines.append(_format_pin(p, x, y, 0))

    # Right side pins: point left (angle 180), placed right of body
    for i, p in enumerate(sides["right"]):
        y = half_h - PIN_SPACING - i * PIN_SPACING
        x = half_w + PIN_LENGTH
        pin_lines.append(_format_pin(p, x, y, 180))

    # Top side pins: point down (angle 270), placed above body
    for i, p in enumerate(sides["top"]):
        x = -half_w + PIN_SPACING + i * PIN_SPACING
        if len(sides["top"]) > 1:
            # Center the pins
            total_span = (len(sides["top"]) - 1) * PIN_SPACING
            x = -total_span / 2 + i * PIN_SPACING
        else:
            x = 0
        y = half_h + PIN_LENGTH
        pin_lines.append(_format_pin(p, x, y, 270))

    # Bottom side pins: point up (angle 90), placed below body
    for i, p in enumerate(sides["bottom"]):
        if len(sides["bottom"]) > 1:
            total_span = (len(sides["bottom"]) - 1) * PIN_SPACING
            x = -total_span / 2 + i * PIN_SPACING
        else:
            x = 0
        y = -half_h - PIN_LENGTH
        pin_lines.append(_format_pin(p, x, y, 90))

    # Calculate property positions
    ref_y = half_h + PIN_LENGTH + 1.27
    val_y = -(half_h + PIN_LENGTH + 1.27)
    if sides["top"]:
        ref_y += 2.54
    if sides["bottom"]:
        val_y -= 2.54

    # Assemble the symbol file
    sym = f"""\
(kicad_symbol_lib
  (version 20231120)
  (generator "pinout_converter")
  (generator_version "1.0")
  (symbol "{safe_name}"
    (pin_names
      (offset 1.016)
    )
    (exclude_from_sim no)
    (in_bom yes)
    (on_board yes)
    (property "Reference" "U"
      (at 0 {ref_y:.4f} 0)
      (effects
        (font
          (size {FONT_SIZE} {FONT_SIZE})
        )
      )
    )
    (property "Value" "{safe_name}"
      (at 0 {val_y:.4f} 0)
      (effects
        (font
          (size {FONT_SIZE} {FONT_SIZE})
        )
      )
    )
    (property "Footprint" ""
      (at 0 0 0)
      (effects
        (font
          (size {FONT_SIZE} {FONT_SIZE})
        )
        (hide yes)
      )
    )
    (property "Datasheet" ""
      (at 0 0 0)
      (effects
        (font
          (size {FONT_SIZE} {FONT_SIZE})
        )
        (hide yes)
      )
    )
    (symbol "{safe_name}_0_1"
      (rectangle
        (start {-half_w:.4f} {half_h:.4f})
        (end {half_w:.4f} {-half_h:.4f})
        (stroke
          (width 0.254)
          (type default)
        )
        (fill
          (type background)
        )
      )
    )
    (symbol "{safe_name}_1_1"
{chr(10).join(pin_lines)}
    )
  )
)
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sym)

    print(f"\nKiCad symbol written to: {output_path}")
    print(f"  Component: {safe_name}")
    print(f"  Pins: {len(pins)}")
    print(f"  Body: {body_width:.2f} x {body_height:.2f} mm")


def _round_to_grid(value: float, grid: float = 2.54) -> float:
    """Round a value up to the nearest grid increment."""
    import math

    return math.ceil(value / grid) * grid


def _format_pin(pin: Pin, x: float, y: float, angle: int) -> str:
    """Format a single pin as a KiCad S-expression string."""
    etype = KICAD_PIN_TYPE_MAP.get(pin.electrical_type, "unspecified")
    escaped_name = pin.name.replace('"', '\\"')
    escaped_num = pin.number.replace('"', '\\"')
    return f"""\
      (pin {etype} line
        (at {x:.4f} {y:.4f} {angle})
        (length {PIN_LENGTH})
        (name "{escaped_name}"
          (effects
            (font
              (size {FONT_SIZE} {FONT_SIZE})
            )
          )
        )
        (number "{escaped_num}"
          (effects
            (font
              (size {FONT_SIZE} {FONT_SIZE})
            )
          )
        )
      )"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert a pinout diagram image to a KiCad .kicad_sym symbol file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pinout_converter.py chip_pinout.png
  python pinout_converter.py datasheet.pdf --page 3
  python pinout_converter.py pinout.jpg --output my_chip.kicad_sym --name ATmega328P
""",
    )
    parser.add_argument("input", help="Path to a pinout diagram image (PNG/JPG) or PDF")
    parser.add_argument(
        "-o", "--output", help="Output .kicad_sym file path (default: <component_name>.kicad_sym)"
    )
    parser.add_argument(
        "-n", "--name", help="Override the component name (default: auto-detected)"
    )
    parser.add_argument(
        "-p", "--page", type=int, default=None, help="PDF page number to use (1-indexed)"
    )
    parser.add_argument(
        "--api-key", help="Anthropic API key (default: uses ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip interactive review and generate symbol directly",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: File not found: {input_path}")

    # Determine if PDF or image
    is_pdf = input_path.suffix.lower() == ".pdf"

    if is_pdf:
        page_count = get_pdf_page_count(str(input_path))
        print(f"PDF loaded: {input_path.name} ({page_count} pages)")

        if args.page is not None:
            page_num = args.page
        else:
            page_num = int(input(f"Enter page number (1-{page_count}): ").strip())

        image_b64, media_type = extract_pdf_page_as_image(str(input_path), page_num)
        print(f"Extracted page {page_num} as image.")
    else:
        image_b64, media_type = load_image_as_base64(str(input_path))
        print(f"Image loaded: {input_path.name}")

    # Extract pins using Claude Vision
    component_name, pins = extract_pins_with_claude(
        image_b64, media_type, api_key=args.api_key
    )

    if not pins:
        print("No pins were extracted from the image.")
        print("The image may not contain a recognizable pinout diagram.")
        sys.exit(1)

    # Override component name if specified
    if args.name:
        component_name = args.name

    # Interactive review
    if not args.no_review:
        component_name, pins = interactive_review(component_name, pins)
    else:
        print(f"\nComponent: {component_name}")
        print(f"Extracted {len(pins)} pins (skipping review)")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        safe = _sanitize_name(component_name)
        output_path = f"{safe}.kicad_sym"

    # Generate symbol
    generate_kicad_symbol(component_name, pins, output_path)
    print("\nDone! Open the .kicad_sym file in KiCad Symbol Editor to verify.")


if __name__ == "__main__":
    main()
