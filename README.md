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

## Web app
Launch the website to enter a TIC target ID or upload a local light-curve file, then view the result summary and plots in the browser:

```powershell
& ".venv/Scripts/python.exe" app.py
```

You can also provide an optional stellar radius to improve the planet-radius estimate. For TESS targets, the app will try to pull the radius from the TIC catalog automatically.

## First utility
Preview a local light-curve file or download a TESS target by TIC ID:

```powershell
& ".venv/Scripts/python.exe" preview_dataset.py --path path/to/file.csv
& ".venv/Scripts/python.exe" preview_dataset.py --target-id 123456789 --sector 1
```

The preview will print a readable summary and save a plot in `artifacts/dataset_preview.png` by default.

## Project Setup (For developers)
- Clone the repository and *cd* to cloned folder.
1. Backend Setup
    - *cd* in backend folder.
    - Create a virtual environment `python -m venv myenv`
    - Activate the virtual environment `.\myenv\Scripts\activate`
    - Install requirements `pip install -r requirements.txt`
    - Start app.py (web server) `python app.py` or `.\myenv\Scripts\python.exe app.py`

2. Start frontend
    - *cd* in frontend folder.
    - run `npm i` to install dependencies.
    - start project `npm run start`
    Note: Proxy is setup in package.json so it will connect frontend to backend during development automatically.
    Note: Visit the url given by `npm run start` not the one given my running app.py since it's development level.

3. Build/Production
    - *cd* in frontend folder.
    - Build the react application `npm run build`
    - *cd* in backend folder.
    - Start app.py (web server) `python app.py` or `.\myenv\Scripts\python.exe app.py`
    - Visit the url given by app.py