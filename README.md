# Exoplanet Detection Pipeline

This project builds a modular AI pipeline to detect and classify exoplanet transit-like signals in noisy TESS light curves.

## Current focus
- Dataset inspection and preview
- Light-curve ingestion from TESS or local files
- Clean modular structure by responsibility

## Transit search
Search for periodic dips and estimate the strongest candidate period using BLS:

```powershell
& ".venv/Scripts/python.exe" search_transits.py --target-id 123456789 --all-sectors --show
& ".venv/Scripts/python.exe" search_transits.py --path .\your_lightcurve.fits --show
```

## Desktop app
Launch the GUI to enter a TIC target ID or browse a local light-curve file, then view the result summary and plots in the app:

```powershell
& ".venv/Scripts/python.exe" exoplanet_gui.py
```

## First utility
Preview a local light-curve file or download a TESS target by TIC ID:

```powershell
& ".venv/Scripts/python.exe" preview_dataset.py --path path/to/file.csv
& ".venv/Scripts/python.exe" preview_dataset.py --target-id 123456789 --sector 1
```

The preview will print a readable summary and save a plot in `artifacts/dataset_preview.png` by default.
