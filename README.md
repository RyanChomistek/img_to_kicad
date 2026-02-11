# img_to_kicad

CLI tool that converts a pinout diagram image (or PDF page) into KiCad 6+ symbol (`.kicad_sym`) and footprint (`.kicad_mod`) files using Claude Vision API.

## How it works

1. Feed it an image or PDF of a pinout diagram
2. Claude Vision extracts pin names, numbers, positions, electrical types, and infers the physical package type and dimensions
3. Review and edit the extracted data interactively in your terminal
4. The tool generates both a `.kicad_sym` symbol and a `.kicad_mod` footprint, ready to open in KiCad

## Requirements

- Python 3.8+
- An [Anthropic API key](https://console.anthropic.com/) (set `ANTHROPIC_API_KEY` env var or pass `--api-key`)

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `anthropic`, `Pillow`, `PyMuPDF`

## Usage

### Basic

```bash
# From an image
python pinout_converter.py chip_pinout.png

# From a PDF (specific page)
python pinout_converter.py datasheet.pdf --page 3
```

### Naming and output paths

```bash
# Override the component name (used for both symbol and footprint)
python pinout_converter.py pinout.jpg --name ATmega328P

# Custom output paths
python pinout_converter.py pinout.jpg -o my_chip.kicad_sym -fo my_chip.kicad_mod
```

### Skipping steps

```bash
# Skip interactive review (accept Claude's extraction as-is)
python pinout_converter.py chip.png --no-review

# Generate symbol only, no footprint
python pinout_converter.py chip.png --no-footprint
```

### Footprint overrides

Override any footprint parameter from the command line. These are applied on top of Claude's inference and can still be edited in interactive review.

```bash
# Force SOIC package with specific dimensions
python pinout_converter.py chip.png --package-type soic --pin-pitch 1.27 --row-spacing 5.4

# Set pad geometry
python pinout_converter.py chip.png --pad-width 0.6 --pad-height 1.5 --pad-shape rect --pad-type smd

# Through-hole with custom drill
python pinout_converter.py chip.png --package-type dip --pad-type thru_hole --drill-size 0.8

# Enable thermal pad (common for QFN)
python pinout_converter.py chip.png --package-type qfn --thermal-pad --thermal-pad-width 3.5 --thermal-pad-height 3.5
```

## All options

```
positional arguments:
  input                     Image (PNG/JPG/BMP/GIF/WebP) or PDF file

optional arguments:
  -o, --output              Symbol output path (default: <name>.kicad_sym)
  -fo, --footprint-output   Footprint output path (default: <name>.kicad_mod)
  -n, --name                Override component name
  -p, --page                PDF page number (1-indexed)
  --api-key                 Anthropic API key
  --no-review               Skip interactive review
  --no-footprint            Skip footprint generation

footprint overrides:
  --package-type            dip, soic, ssop, sop, tssop, qfp, tqfp, lqfp,
                            qfn, dfn, bga, sot, other
  --pin-pitch               Pin pitch in mm
  --pad-width               Pad width in mm
  --pad-height              Pad height in mm
  --row-spacing             Row spacing (center-to-center) in mm
  --body-width              Body width in mm
  --body-height             Body height in mm
  --pad-shape               rect, oval, circle, roundrect
  --pad-type                smd, thru_hole
  --drill-size              Drill diameter in mm (thru_hole only)
  --thermal-pad             Enable thermal/exposed pad
  --no-thermal-pad          Disable thermal/exposed pad
  --thermal-pad-width       Thermal pad width in mm
  --thermal-pad-height      Thermal pad height in mm
```

## Interactive review

When running without `--no-review`, you get two review stages:

**Pin review** -- edit, add, delete, or reorder pins:
```
Idx   Number   Name                 Side     Pos   Type
-----------------------------------------------------------------
0     1        VCC                  left     1     power_in
1     2        GND                  left     2     power_in
...

Options:
  [a] Accept and generate symbol
  [e] Edit a pin
  [d] Delete a pin
  [n] Add a new pin
  [r] Rename component
  [p] Print pin table again
  [q] Quit without saving
```

**Footprint review** -- adjust package parameters before generating:
```
  Package type:      SOIC
  Pin count:         8
  Pin pitch:         1.27 mm
  Pad size:          0.6 x 1.5 mm
  Row spacing:       5.4 mm
  ...

Footprint options:
  [a] Accept footprint parameters
  [e] Edit a parameter
  [t] Toggle thermal pad on/off
  [p] Print parameters again
  [s] Skip footprint generation
```

## Supported packages

| Type | Layout | Pad type |
|------|--------|----------|
| DIP, SOIC, SSOP, SOP, TSSOP, SOT, DFN | Dual-row | thru_hole or smd |
| QFP, TQFP, LQFP, QFN | Quad (4-side) | smd |
| BGA | Grid (alphanumeric) | smd |

## Output details

Generated footprints include:
- Pads with correct numbering and pin 1 marker
- Silkscreen outline (F.SilkS)
- Fabrication layer outline (F.Fab)
- Courtyard (F.CrtYd) with 0.25 mm clearance
- Reference and Value text
- Optional thermal/exposed pad

The symbol's `Footprint` property is automatically linked to the generated footprint name.
