# Single-image count implementation plan

**Goal:** A Windows bench tool with explicit 14 mm / 12+14 mm modes, one-image input, live class totals, manual correction and original-size annotated PNG export.

**Architecture:** Keep detector v2 and the accepted 1200 px viewer rendering path. `vibrotrack count` with no image starts a loopback-only local page that accepts one image per request; the existing file/bundle path shares the detector and payload. Neighbours are opt-in for the CLI only. A preset is scoped to the present apparatus/image scale, not a guarantee of general polydisperse detection.

**Stack:** Python stdlib HTTP server, vibrotrack/OpenCV/Pillow, existing vanilla HTML/Canvas.

1. `count.py`: implement named presets, bytes decoding and a shared detector helper, add class metadata and safe JSON embedding; verify equal results for CLI and upload where settings match, no implicit neighbours, empty detection and color input.
2. `count_app.py`, `cli.py`, `count_viewer.html`: explicit mode chooser, bounded same-origin loopback upload, accurate count after add/delete/undo, input errors and PNG export. Keep passive bundles working. Test server round-trip and offline DOM/Canvas call sequence under Node; no browser automation in this preset.
3. `README.md`, `tests/test_count.py`: replace the misleading cross-frame table, verify using one source frame for both modes and report per-class counts without claiming accuracy from totals. Run `python -m pytest -q`, JS smoke tests and `git diff --check`.
