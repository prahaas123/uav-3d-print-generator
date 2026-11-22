# rc_print_parts.py

Builds print-ready parts for a conventional pod-and-boom RC plane.

**Input:** the OML as VSP-style planform numbers (`CONFIG` in the script). The fuselage is not part of the OML, so the script designs it from its own station table.

**Units:** mm, degrees. **Aircraft axes:** x aft, y right, z up (same as the OML plot).

## Parts made

| Group | Parts | Notes |
| ------- | ------- | ------- |
| **Wing** | 2–3 panels per side | Single-wall vase-mode (spiralize) parts with a ±45° slot lattice (v5 style), a continuous CF spar bore and alignment-pin bores. |
| **Tail** | Mount (normal print), left HT, right HT, VT (vase mode) | The HT has one straight CF rod through the mount and both halves. The VT has a CF rod that plugs into the mount. |
| **Fuselage** | Nose (firewall + spigot), wingbox (wing saddle, rubber-band dowels, boom socket) | Normal prints. |

## How the vase-mode structure works

This uses the same idea as the UAV 2 v5 RWing.

Each wing and tail part starts as a solid OML. Thin slots (`SLOT_WIDTH`) are cut from the top skin down to just above the camber surface, and from the bottom skin up to just below it, leaving a `SLOT_GAP` neck.

In spiral-vase mode the single wall runs into every slot and back out:

- The two passes in each slot fuse into a double-thickness web.
- The passes from the top and bottom slots fuse across the neck.
- Every rod bore is joined to the bottom skin by a through-slit, so each layer is **one closed contour** and the part prints in one continuous move with no retracts.

> **Rule:** keep `LINE_WIDTH < SLOT_GAP < 2 * LINE_WIDTH`.

## Outputs

Written to `--out` (default `./rc_parts`):

| Path | Contents |
| ---- | -------- |
| `step/<part>.step` | Parts in aircraft coordinates (open in SolidWorks, or build an assembly) |
| `step/assembly.step` | Everything, positioned |
| `stl/<part>.stl` | Parts in print orientation (z = print direction, sitting on z = 0) |
| `summary.txt` | Rod lengths/positions, print orientation, mass estimates, warnings |
| `preview.png` | 4-view render (if matplotlib is installed) |

## Usage

Install dependencies:

```bash
pip install cadquery numpy trimesh networkx matplotlib
```

Run:

```bash
python rc_print_parts.py [--out DIR] [--check] [--no-stl] [--preview]
```

| Flag | Description |
| ---- | ----------- |
| `--out DIR` | Output directory (default `./rc_parts`) |
| `--check` | Slices every vase STL at 0.2 mm and confirms each layer is a single loop |
| `--no-stl` | Skip STL export |
| `--preview` | Render `preview.png` |

A full run takes about 5–8 minutes. The slot booleans on the inboard wing panels take most of that time.
