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


VALID_PACKAGE_TYPES = (
    "DIP", "SOIC", "SSOP", "SOP", "TSSOP",
    "QFP", "TQFP", "LQFP",
    "QFN", "DFN",
    "BGA",
    "SOT",
    "OTHER",
)

VALID_PAD_SHAPES = ("rect", "oval", "circle", "roundrect")
VALID_PAD_TYPES = ("smd", "thru_hole")

# Dual-row package types (pins on left and right sides)
DUAL_ROW_TYPES = ("DIP", "SOIC", "SSOP", "SOP", "TSSOP", "SOT", "DFN")
# Quad package types (pins on all four sides)
QUAD_TYPES = ("QFP", "TQFP", "LQFP", "QFN")


@dataclass
class Package:
    package_type: str = "DIP"
    pin_count: int = 8
    pin_pitch: float = 2.54
    pad_width: float = 1.6
    pad_height: float = 1.6
    row_spacing: float = 7.62
    body_width: float = 6.35
    body_height: float = 9.27
    pad_shape: str = "oval"
    pad_type: str = "thru_hole"
    drill_size: float = 1.0
    thermal_pad: bool = False
    thermal_pad_width: float = 0.0
    thermal_pad_height: float = 0.0

    def __post_init__(self):
        if self.package_type not in VALID_PACKAGE_TYPES:
            self.package_type = "OTHER"
        if self.pad_shape not in VALID_PAD_SHAPES:
            self.pad_shape = "oval"
        if self.pad_type not in VALID_PAD_TYPES:
            self.pad_type = "thru_hole"


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
You are analyzing an image of an IC/component pinout diagram. Extract ALL pins shown in the diagram
AND infer the physical package type and dimensions.

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
  ],
  "package": {
    "package_type": "DIP",
    "pin_count": 8,
    "pin_pitch": 2.54,
    "pad_width": 1.6,
    "pad_height": 1.6,
    "row_spacing": 7.62,
    "body_width": 6.35,
    "body_height": 9.27,
    "pad_shape": "oval",
    "pad_type": "thru_hole",
    "drill_size": 1.0,
    "thermal_pad": false,
    "thermal_pad_width": 0.0,
    "thermal_pad_height": 0.0
  }
}

Pin rules:
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

Package rules:
- "package_type": One of "DIP", "SOIC", "SSOP", "SOP", "TSSOP", "QFP", "TQFP", "LQFP",
  "QFN", "DFN", "BGA", "SOT", "OTHER". Infer from visual cues:
  - DIP: through-hole, wide body, notch at top
  - SOIC/SSOP/SOP/TSSOP: narrow body, gull-wing SMD leads on two sides
  - QFP/TQFP/LQFP: gull-wing leads on all four sides
  - QFN/DFN: no-lead, pads underneath on sides, often has exposed thermal pad
  - BGA: ball grid array underneath
  - SOT: small outline transistor (3-8 pins)
- "pin_pitch": mm between adjacent pad centers. Common values:
  DIP=2.54, SOIC=1.27, SSOP/TSSOP=0.65, QFP=0.5/0.65/0.8, QFN=0.5/0.65
- "pad_width"/"pad_height": pad dimensions in mm
- "row_spacing": center-to-center distance between opposing pad rows in mm
- "body_width"/"body_height": silkscreen body outline in mm
- "pad_shape": "rect", "oval", "circle", or "roundrect"
- "pad_type": "smd" for surface mount, "thru_hole" for through-hole (DIP)
- "drill_size": drill diameter in mm (only for thru_hole, 0 for SMD)
- "thermal_pad": true if the package has an exposed/thermal pad (common in QFN)
- "thermal_pad_width"/"thermal_pad_height": size of thermal pad in mm (0 if none)

Important:
- Extract EVERY pin visible in the diagram. Do not skip any.
- The position values are CRITICAL: they must match the visual layout in the image.
  Look carefully at the vertical/horizontal arrangement of each pin on its side.
- If the diagram shows a chip from the top, left side pins typically have low numbers.
- Use standard package dimensions when possible. If uncertain, estimate conservatively.
- Return ONLY valid JSON. No explanation, no markdown code fences.
"""


def _parse_package(data: dict, pin_count: int) -> Package:
    """Parse package dict from Claude response into a Package instance with defaults."""
    pkg_data = data.get("package", {})
    return Package(
        package_type=pkg_data.get("package_type", "DIP"),
        pin_count=pin_count,
        pin_pitch=float(pkg_data.get("pin_pitch", 2.54)),
        pad_width=float(pkg_data.get("pad_width", 1.6)),
        pad_height=float(pkg_data.get("pad_height", 1.6)),
        row_spacing=float(pkg_data.get("row_spacing", 7.62)),
        body_width=float(pkg_data.get("body_width", 6.35)),
        body_height=float(pkg_data.get("body_height", 9.27)),
        pad_shape=pkg_data.get("pad_shape", "oval"),
        pad_type=pkg_data.get("pad_type", "thru_hole"),
        drill_size=float(pkg_data.get("drill_size", 1.0)),
        thermal_pad=bool(pkg_data.get("thermal_pad", False)),
        thermal_pad_width=float(pkg_data.get("thermal_pad_width", 0.0)),
        thermal_pad_height=float(pkg_data.get("thermal_pad_height", 0.0)),
    )


def extract_pins_with_claude(
    image_b64: str, media_type: str, api_key: str | None = None
) -> tuple[str, list[Pin], Package]:
    """Send image to Claude Vision API and extract pin + package information."""
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

    package = _parse_package(data, len(pins))

    return component_name, pins, package


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


def print_package_table(pkg: Package):
    """Print package parameters in a readable format."""
    print(f"\n  Package type:      {pkg.package_type}")
    print(f"  Pin count:         {pkg.pin_count}")
    print(f"  Pin pitch:         {pkg.pin_pitch} mm")
    print(f"  Pad size:          {pkg.pad_width} x {pkg.pad_height} mm")
    print(f"  Row spacing:       {pkg.row_spacing} mm")
    print(f"  Body size:         {pkg.body_width} x {pkg.body_height} mm")
    print(f"  Pad shape:         {pkg.pad_shape}")
    print(f"  Pad type:          {pkg.pad_type}")
    if pkg.pad_type == "thru_hole":
        print(f"  Drill size:        {pkg.drill_size} mm")
    print(f"  Thermal pad:       {'Yes' if pkg.thermal_pad else 'No'}")
    if pkg.thermal_pad:
        print(f"  Thermal pad size:  {pkg.thermal_pad_width} x {pkg.thermal_pad_height} mm")
    print()


PACKAGE_FIELD_NAMES = [
    ("package_type", "Package type", str),
    ("pin_pitch", "Pin pitch (mm)", float),
    ("pad_width", "Pad width (mm)", float),
    ("pad_height", "Pad height (mm)", float),
    ("row_spacing", "Row spacing (mm)", float),
    ("body_width", "Body width (mm)", float),
    ("body_height", "Body height (mm)", float),
    ("pad_shape", "Pad shape (rect/oval/circle/roundrect)", str),
    ("pad_type", "Pad type (smd/thru_hole)", str),
    ("drill_size", "Drill size (mm)", float),
]


def interactive_footprint_review(pkg: Package) -> Package:
    """Let the user review and edit extracted package parameters."""
    print("\nFootprint parameters (extracted from image):")
    print_package_table(pkg)

    while True:
        print("Footprint options:")
        print("  [a] Accept footprint parameters")
        print("  [e] Edit a parameter")
        print("  [t] Toggle thermal pad on/off")
        print("  [p] Print parameters again")
        print("  [s] Skip footprint generation")

        choice = input("\nChoice: ").strip().lower()

        if choice == "a":
            return pkg

        elif choice == "e":
            print("\nEditable fields:")
            for i, (field, label, _) in enumerate(PACKAGE_FIELD_NAMES):
                val = getattr(pkg, field)
                print(f"  [{i}] {label}: {val}")
            try:
                idx = int(input("Field number to edit: "))
                if idx < 0 or idx >= len(PACKAGE_FIELD_NAMES):
                    print(f"Invalid index. Must be 0-{len(PACKAGE_FIELD_NAMES)-1}")
                    continue
            except ValueError:
                print("Invalid input.")
                continue

            field, label, typ = PACKAGE_FIELD_NAMES[idx]
            current = getattr(pkg, field)
            new_val = input(f"  {label} [{current}]: ").strip()
            if new_val:
                try:
                    converted = typ(new_val)
                    # Validate specific fields
                    if field == "package_type":
                        converted = converted.upper()
                        if converted not in VALID_PACKAGE_TYPES:
                            print(f"  Invalid package type. Valid: {', '.join(VALID_PACKAGE_TYPES)}")
                            continue
                    elif field == "pad_shape" and converted not in VALID_PAD_SHAPES:
                        print(f"  Invalid pad shape. Valid: {', '.join(VALID_PAD_SHAPES)}")
                        continue
                    elif field == "pad_type" and converted not in VALID_PAD_TYPES:
                        print(f"  Invalid pad type. Valid: {', '.join(VALID_PAD_TYPES)}")
                        continue
                    setattr(pkg, field, converted)
                    print("  Updated.")
                except ValueError:
                    print(f"  Invalid value for {label}.")
            print_package_table(pkg)

        elif choice == "t":
            pkg.thermal_pad = not pkg.thermal_pad
            if pkg.thermal_pad and pkg.thermal_pad_width == 0:
                w = input("  Thermal pad width (mm): ").strip()
                h = input("  Thermal pad height (mm): ").strip()
                try:
                    pkg.thermal_pad_width = float(w) if w else 3.0
                    pkg.thermal_pad_height = float(h) if h else 3.0
                except ValueError:
                    pkg.thermal_pad_width = 3.0
                    pkg.thermal_pad_height = 3.0
            print(f"  Thermal pad: {'On' if pkg.thermal_pad else 'Off'}")
            print_package_table(pkg)

        elif choice == "p":
            print_package_table(pkg)

        elif choice == "s":
            return None

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
    component_name: str, pins: list[Pin], output_path: str, footprint_ref: str = ""
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

    # Body width: must fit the longest left name + longest right name + gap
    PIN_NAME_OFFSET = 1.016  # matches pin_names offset in symbol header
    CHAR_WIDTH = FONT_SIZE * 0.7  # approximate character width at font size
    GAP = 2.54  # minimum gap between left and right pin name text

    left_names = [p.name for p in sides["left"]]
    right_names = [p.name for p in sides["right"]]
    max_left_len = max((len(n) for n in left_names), default=0)
    max_right_len = max((len(n) for n in right_names), default=0)

    # Width needed so left and right pin names don't overlap
    names_width = (max_left_len + max_right_len) * CHAR_WIDTH + PIN_NAME_OFFSET * 2 + GAP
    horiz_width = max(max_horizontal * PIN_SPACING + PIN_SPACING, 5.08)
    body_width = max(names_width, horiz_width, BODY_MIN_WIDTH)
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
    (property "Footprint" "{footprint_ref}"
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
# KiCad footprint generation
# ---------------------------------------------------------------------------

COURTYARD_CLEARANCE = 0.25  # mm
SILKSCREEN_WIDTH = 0.12  # mm
FAB_WIDTH = 0.1  # mm
COURTYARD_WIDTH = 0.05  # mm
FP_FONT_SIZE = 1.0  # mm


def generate_kicad_footprint(
    component_name: str, pkg: Package, output_path: str
):
    """Generate a .kicad_mod footprint file from package parameters."""
    import math

    safe_name = _sanitize_name(component_name)
    lines = []

    def add(text: str):
        lines.append(text)

    add(f'(footprint "{safe_name}"')
    add('  (version 20240108)')
    add('  (generator "pinout_converter")')
    add('  (layer "F.Cu")')
    add(f'  (property "Reference" "REF**"')
    add(f'    (at 0 {-pkg.body_height / 2 - 1.5:.4f} 0)')
    add('    (layer "F.SilkS")')
    add(f'    (effects (font (size {FP_FONT_SIZE} {FP_FONT_SIZE}) (thickness 0.15)))')
    add('  )')
    add(f'  (property "Value" "{safe_name}"')
    add(f'    (at 0 {pkg.body_height / 2 + 1.5:.4f} 0)')
    add('    (layer "F.Fab")')
    add(f'    (effects (font (size {FP_FONT_SIZE} {FP_FONT_SIZE}) (thickness 0.15)))')
    add('  )')

    # Compute pad positions
    pad_entries = []  # list of (number, x, y, angle)

    pkg_upper = pkg.package_type.upper()

    if pkg_upper == "BGA":
        # BGA: grid arrangement with alphanumeric naming
        cols = int(math.ceil(math.sqrt(pkg.pin_count)))
        rows = int(math.ceil(pkg.pin_count / cols))
        total_w = (cols - 1) * pkg.pin_pitch
        total_h = (rows - 1) * pkg.pin_pitch
        count = 0
        for r in range(rows):
            row_letter = chr(ord('A') + r)
            for c in range(cols):
                count += 1
                if count > pkg.pin_count:
                    break
                x = -total_w / 2 + c * pkg.pin_pitch
                y = -total_h / 2 + r * pkg.pin_pitch
                pad_name = f"{row_letter}{c + 1}"
                pad_entries.append((pad_name, x, y, 0))

    elif pkg_upper in QUAD_TYPES:
        # Quad: pins on all 4 sides
        pps = pkg.pin_count // 4  # pins per side
        remainder = pkg.pin_count % 4
        # Distribute remainder to sides: left, bottom, right, top
        side_counts = [pps] * 4
        for i in range(remainder):
            side_counts[i] += 1

        half_rs = pkg.row_spacing / 2
        pin_num = 1

        # Left side: top to bottom
        n = side_counts[0]
        span = (n - 1) * pkg.pin_pitch
        for i in range(n):
            x = -half_rs
            y = -span / 2 + i * pkg.pin_pitch
            pad_entries.append((str(pin_num), x, y, 0))
            pin_num += 1

        # Bottom side: left to right
        n = side_counts[1]
        span = (n - 1) * pkg.pin_pitch
        for i in range(n):
            x = -span / 2 + i * pkg.pin_pitch
            y = half_rs
            pad_entries.append((str(pin_num), x, y, 90))
            pin_num += 1

        # Right side: bottom to top
        n = side_counts[2]
        span = (n - 1) * pkg.pin_pitch
        for i in range(n):
            x = half_rs
            y = span / 2 - i * pkg.pin_pitch
            pad_entries.append((str(pin_num), x, y, 0))
            pin_num += 1

        # Top side: right to left
        n = side_counts[3]
        span = (n - 1) * pkg.pin_pitch
        for i in range(n):
            x = span / 2 - i * pkg.pin_pitch
            y = -half_rs
            pad_entries.append((str(pin_num), x, y, 90))
            pin_num += 1

    else:
        # Dual-row: DIP, SOIC, SSOP, SOP, TSSOP, SOT, DFN, OTHER
        half_count = pkg.pin_count // 2
        half_rs = pkg.row_spacing / 2
        span = (half_count - 1) * pkg.pin_pitch

        # Left column: pins 1..N/2 top to bottom
        for i in range(half_count):
            x = -half_rs
            y = -span / 2 + i * pkg.pin_pitch
            pad_entries.append((str(i + 1), x, y, 0))

        # Right column: pins N/2+1..N bottom to top
        for i in range(half_count):
            x = half_rs
            y = span / 2 - i * pkg.pin_pitch
            pad_entries.append((str(half_count + i + 1), x, y, 0))

        # Handle odd pin (e.g. SOT-23 with 3 pins): extra pin at bottom center
        if pkg.pin_count % 2 == 1:
            x = 0
            y = span / 2 + pkg.pin_pitch
            pad_entries.append((str(pkg.pin_count), x, y, 90))

    # Write pad entries
    layers_smd = '(layers "F.Cu" "F.Paste" "F.Mask")'
    layers_th = '(layers "*.Cu" "*.Mask")'

    for idx, (pad_num, x, y, angle) in enumerate(pad_entries):
        pad_type = pkg.pad_type
        shape = pkg.pad_shape

        # Determine pad width/height based on angle
        if angle == 90:
            pw, ph = pkg.pad_height, pkg.pad_width
        else:
            pw, ph = pkg.pad_width, pkg.pad_height

        layers = layers_th if pad_type == "thru_hole" else layers_smd
        drill_str = f' (drill {pkg.drill_size:.4f})' if pad_type == "thru_hole" else ""

        add(f'  (pad "{pad_num}" {pad_type} {shape}')
        add(f'    (at {x:.4f} {y:.4f})')
        add(f'    (size {pw:.4f} {ph:.4f})')
        add(f'    {layers}{drill_str}')
        add('  )')

    # Thermal pad
    if pkg.thermal_pad and pkg.thermal_pad_width > 0 and pkg.thermal_pad_height > 0:
        add(f'  (pad "" smd rect')
        add(f'    (at 0 0)')
        add(f'    (size {pkg.thermal_pad_width:.4f} {pkg.thermal_pad_height:.4f})')
        add(f'    {layers_smd}')
        add('  )')

    # Silkscreen outline with pin 1 marker
    half_bw = pkg.body_width / 2
    half_bh = pkg.body_height / 2

    # Silkscreen rectangle
    for x1, y1, x2, y2 in [
        (-half_bw, -half_bh, half_bw, -half_bh),   # top
        (half_bw, -half_bh, half_bw, half_bh),      # right
        (half_bw, half_bh, -half_bw, half_bh),      # bottom
        (-half_bw, half_bh, -half_bw, -half_bh),    # left
    ]:
        add(f'  (fp_line (start {x1:.4f} {y1:.4f}) (end {x2:.4f} {y2:.4f})')
        add(f'    (stroke (width {SILKSCREEN_WIDTH}) (type solid)) (layer "F.SilkS"))')

    # Pin 1 marker: small circle on silkscreen near pin 1
    if pad_entries:
        p1_x, p1_y = pad_entries[0][1], pad_entries[0][2]
        marker_x = p1_x + (0.5 if p1_x < 0 else -0.5)
        marker_y = p1_y
        # Clamp marker inside body outline
        marker_x = max(-half_bw + 0.5, min(half_bw - 0.5, marker_x))
        marker_y = max(-half_bh + 0.5, min(half_bh - 0.5, marker_y))
        add(f'  (fp_circle (center {marker_x:.4f} {marker_y:.4f}) (end {marker_x + 0.25:.4f} {marker_y:.4f})')
        add(f'    (stroke (width {SILKSCREEN_WIDTH}) (type solid)) (fill solid) (layer "F.SilkS"))')

    # Fabrication layer outline
    for x1, y1, x2, y2 in [
        (-half_bw, -half_bh, half_bw, -half_bh),
        (half_bw, -half_bh, half_bw, half_bh),
        (half_bw, half_bh, -half_bw, half_bh),
        (-half_bw, half_bh, -half_bw, -half_bh),
    ]:
        add(f'  (fp_line (start {x1:.4f} {y1:.4f}) (end {x2:.4f} {y2:.4f})')
        add(f'    (stroke (width {FAB_WIDTH}) (type solid)) (layer "F.Fab"))')

    # Courtyard
    cx = half_bw + COURTYARD_CLEARANCE
    cy = half_bh + COURTYARD_CLEARANCE
    # Extend courtyard to cover pads
    if pad_entries:
        all_x = [abs(e[1]) + pkg.pad_width / 2 for e in pad_entries]
        all_y = [abs(e[2]) + pkg.pad_height / 2 for e in pad_entries]
        cx = max(cx, max(all_x) + COURTYARD_CLEARANCE)
        cy = max(cy, max(all_y) + COURTYARD_CLEARANCE)

    for x1, y1, x2, y2 in [
        (-cx, -cy, cx, -cy),
        (cx, -cy, cx, cy),
        (cx, cy, -cx, cy),
        (-cx, cy, -cx, -cy),
    ]:
        add(f'  (fp_line (start {x1:.4f} {y1:.4f}) (end {x2:.4f} {y2:.4f})')
        add(f'    (stroke (width {COURTYARD_WIDTH}) (type solid)) (layer "F.CrtYd"))')

    add(')')

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nKiCad footprint written to: {output_path}")
    print(f"  Package: {pkg.package_type}-{pkg.pin_count}")
    print(f"  Pads: {len(pad_entries)}")
    print(f"  Body: {pkg.body_width:.2f} x {pkg.body_height:.2f} mm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert a pinout diagram image to KiCad .kicad_sym and .kicad_mod files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pinout_converter.py chip_pinout.png
  python pinout_converter.py datasheet.pdf --page 3
  python pinout_converter.py pinout.jpg --output my_chip.kicad_sym --name ATmega328P
  python pinout_converter.py chip.png --no-footprint
  python pinout_converter.py chip.png -fo custom_footprint.kicad_mod
""",
    )
    parser.add_argument("input", help="Path to a pinout diagram image (PNG/JPG) or PDF")
    parser.add_argument(
        "-o", "--output", help="Output .kicad_sym file path (default: <component_name>.kicad_sym)"
    )
    parser.add_argument(
        "-fo", "--footprint-output",
        help="Output .kicad_mod file path (default: <component_name>.kicad_mod)",
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
        help="Skip interactive review and generate directly",
    )
    parser.add_argument(
        "--no-footprint",
        action="store_true",
        help="Skip footprint (.kicad_mod) generation",
    )

    # Footprint parameter overrides
    fp_group = parser.add_argument_group("footprint overrides",
        "Override extracted footprint parameters (applied on top of Claude's inference)")
    fp_group.add_argument(
        "--package-type", choices=[t.lower() for t in VALID_PACKAGE_TYPES],
        help="Package type (e.g. dip, soic, qfp, qfn, bga)",
    )
    fp_group.add_argument("--pin-pitch", type=float, help="Pin pitch in mm")
    fp_group.add_argument("--pad-width", type=float, help="Pad width in mm")
    fp_group.add_argument("--pad-height", type=float, help="Pad height in mm")
    fp_group.add_argument("--row-spacing", type=float, help="Row spacing (center-to-center) in mm")
    fp_group.add_argument("--body-width", type=float, help="Body width in mm")
    fp_group.add_argument("--body-height", type=float, help="Body height in mm")
    fp_group.add_argument(
        "--pad-shape", choices=VALID_PAD_SHAPES,
        help="Pad shape (rect, oval, circle, roundrect)",
    )
    fp_group.add_argument(
        "--pad-type", choices=VALID_PAD_TYPES,
        help="Pad type (smd, thru_hole)",
    )
    fp_group.add_argument("--drill-size", type=float, help="Drill diameter in mm (thru_hole only)")
    fp_group.add_argument(
        "--thermal-pad", action="store_true", default=None,
        help="Enable thermal/exposed pad",
    )
    fp_group.add_argument(
        "--no-thermal-pad", action="store_true", default=None,
        help="Disable thermal/exposed pad",
    )
    fp_group.add_argument("--thermal-pad-width", type=float, help="Thermal pad width in mm")
    fp_group.add_argument("--thermal-pad-height", type=float, help="Thermal pad height in mm")

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

    # Extract pins and package using Claude Vision
    component_name, pins, package = extract_pins_with_claude(
        image_b64, media_type, api_key=args.api_key
    )

    if not pins:
        print("No pins were extracted from the image.")
        print("The image may not contain a recognizable pinout diagram.")
        sys.exit(1)

    # Override component name if specified
    if args.name:
        component_name = args.name

    # Apply CLI footprint overrides
    if args.package_type is not None:
        package.package_type = args.package_type.upper()
    if args.pin_pitch is not None:
        package.pin_pitch = args.pin_pitch
    if args.pad_width is not None:
        package.pad_width = args.pad_width
    if args.pad_height is not None:
        package.pad_height = args.pad_height
    if args.row_spacing is not None:
        package.row_spacing = args.row_spacing
    if args.body_width is not None:
        package.body_width = args.body_width
    if args.body_height is not None:
        package.body_height = args.body_height
    if args.pad_shape is not None:
        package.pad_shape = args.pad_shape
    if args.pad_type is not None:
        package.pad_type = args.pad_type
    if args.drill_size is not None:
        package.drill_size = args.drill_size
    if args.thermal_pad:
        package.thermal_pad = True
    if args.no_thermal_pad:
        package.thermal_pad = False
    if args.thermal_pad_width is not None:
        package.thermal_pad_width = args.thermal_pad_width
    if args.thermal_pad_height is not None:
        package.thermal_pad_height = args.thermal_pad_height

    # Interactive review of pins
    if not args.no_review:
        component_name, pins = interactive_review(component_name, pins)
    else:
        print(f"\nComponent: {component_name}")
        print(f"Extracted {len(pins)} pins (skipping review)")

    # Interactive review of footprint parameters
    generate_fp = not args.no_footprint
    if generate_fp and not args.no_review:
        result = interactive_footprint_review(package)
        if result is None:
            generate_fp = False
        else:
            package = result

    safe = _sanitize_name(component_name)

    # Determine footprint output path and reference
    fp_ref = ""
    if generate_fp:
        if args.footprint_output:
            fp_output_path = args.footprint_output
        else:
            fp_output_path = f"{safe}.kicad_mod"
        fp_ref = safe

    # Determine symbol output path
    if args.output:
        sym_output_path = args.output
    else:
        sym_output_path = f"{safe}.kicad_sym"

    # Generate symbol (with footprint reference if applicable)
    generate_kicad_symbol(component_name, pins, sym_output_path, footprint_ref=fp_ref)

    # Generate footprint
    if generate_fp:
        generate_kicad_footprint(component_name, package, fp_output_path)

    print("\nDone! Open the files in KiCad to verify.")


if __name__ == "__main__":
    main()
