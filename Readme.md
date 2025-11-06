# DigiFlash Archiver 🖥️✨ ('80s Edition)

Quiet, boring, reliable digitization tools for turning paper mountains into searchable archives. '80s vibes, modern under the hood—no drama, just done. Pixel art logo for that retro kick.

## What It Works
- **Scan & Feed**: Drop stacks into your feeder; it chugs 'em at 20-40 ppm (even on '50s faded ink).
- **Auto-Magic**: OCR, naming, and tagging—date, dept, keywords like "1958-Munich-Fishery-Log-045".
- **QC Cousin**: Flags blurry pages or misfeeds before they bite you.
- **Silent Backup**: Hey, accidents happen. This mirrors everything to your drive (or cloud) without a peep.
- **Output**: PDF/A files, ready for gov portals. No folders mess—just one big, labeled pile.

From paper purgatory to pixel paradise. Teal screens and pixel fonts because Gen X nostalgia sells itself.

## Getting Started
1. **Gear Up**: Grab a Fujitsu ScanSnap ($400 CAD) or rent one ($200/week). Laptop with 8GB RAM does fine.
2. **Install**: Clone this repo, `pip install tesseract-ocr pillow scikit-image`.
3. **Run It**: `python scripts/digi_flash.py --folder /path/to/scans --output /path/to/archive`. Walk away—files get handled in 5 mins for 100 pages.
4. **Test**: Throw in 100 old pages—watch it name 'em, check 'em, back 'em up.

Demo in `visuals/demo.gif` (pixel art scanner in action—coming soon).

## The '80s Nod
Teal and magenta pixels, that satisfying *beep* when a stack finishes. For the era of floppy disks and dial-up dreams, but with today's quiet efficiency.

## Roadmap (Quiet Ambitions)
- v1.0: Scan + tag (done).
- v1.1: Auto-flags for staples/jams + email alerts.
- v2.0: Decentralized mirror (because backups should be yours too).

Fork it, tweak it, make it yours. No IP dragons here—just open-source vibes.

[Demo GIF coming soon—pixel art scanner in action.]
