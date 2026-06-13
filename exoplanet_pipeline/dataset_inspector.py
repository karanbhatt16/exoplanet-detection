from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from lightkurve import LightCurve, search_lightcurve


@dataclass
class DatasetPreview:
    """Structured preview information for a light-curve dataset."""

    source: str
    summary: dict[str, Any]
    dataframe: pd.DataFrame | None = None
    lightcurve: LightCurve | None = None
    notes: list[str] = field(default_factory=list)


class DatasetInspector:
    """Inspect and visualize local or remote TESS light-curve data."""

    def __init__(self, target_id: str | None = None, sector: int | None = None):
        self.target_id = target_id
        self.sector = sector

    def load_local(self, path: str | Path) -> DatasetPreview:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        suffix = file_path.suffix.lower()
        notes: list[str] = []

        if suffix in {".csv", ".txt"}:
            dataframe = pd.read_csv(file_path)
            summary = self._summarize_dataframe(dataframe, source=file_path.name)
            return DatasetPreview(source=str(file_path), summary=summary, dataframe=dataframe, notes=notes)

        if suffix in {".parquet"}:
            dataframe = pd.read_parquet(file_path)
            summary = self._summarize_dataframe(dataframe, source=file_path.name)
            return DatasetPreview(source=str(file_path), summary=summary, dataframe=dataframe, notes=notes)

        if suffix in {".fits", ".fit", ".lc"}:
            lightcurve = self._read_lightcurve_file(file_path)
            summary = self._summarize_lightcurve(lightcurve, source=file_path.name)
            notes.append("Loaded as a light-curve file and plotted using time versus normalized flux.")
            return DatasetPreview(source=str(file_path), summary=summary, lightcurve=lightcurve, notes=notes)

        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    def load_remote(self) -> DatasetPreview:
        if not self.target_id:
            raise ValueError("target_id is required for remote download")

        query_target = self.target_id if str(self.target_id).upper().startswith("TIC") else f"TIC {self.target_id}"
        search = search_lightcurve(query_target, mission="TESS", sector=self.sector)
        used_sector_fallback = False
        if len(search) == 0 and self.sector is not None:
            search = search_lightcurve(query_target, mission="TESS")
            used_sector_fallback = True
        if len(search) == 0:
            raise RuntimeError(
                f"No TESS light curves found for {query_target}. "
                "Check that the TIC ID exists, try a different sector, or preview a local FITS/CSV file instead."
            )

        lightcurve = search.download()
        if isinstance(lightcurve, list):
            lightcurve = lightcurve[0]

        if not isinstance(lightcurve, LightCurve):
            raise RuntimeError("Could not download a single LightCurve object")

        summary = self._summarize_lightcurve(lightcurve, source=query_target)
        notes = ["Downloaded with lightkurve from MAST."]
        if self.sector is not None and used_sector_fallback:
            notes.append(f"Requested sector {self.sector} had no direct match, so the lookup fell back to any available TESS sector.")
        return DatasetPreview(source=query_target, summary=summary, lightcurve=lightcurve, notes=notes)

    def preview_table(self, dataframe: pd.DataFrame, max_rows: int = 8) -> str:
        if dataframe.empty:
            return "Dataframe is empty."
        return dataframe.head(max_rows).to_string(index=False)

    def plot_preview(
        self,
        preview: DatasetPreview,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        fig, ax = plt.subplots(figsize=(11, 5))

        if preview.lightcurve is not None:
            lightcurve = preview.lightcurve
            time = np.asarray(lightcurve.time.value if hasattr(lightcurve.time, "value") else lightcurve.time)
            flux = np.asarray(lightcurve.flux.value if hasattr(lightcurve.flux, "value") else lightcurve.flux)
            flux_err = None
            if getattr(lightcurve, "flux_err", None) is not None:
                flux_err = np.asarray(
                    lightcurve.flux_err.value if hasattr(lightcurve.flux_err, "value") else lightcurve.flux_err
                )
            baseline = np.nanmedian(flux)
            normalized_flux = flux / baseline if baseline not in (0, np.nan) else flux
            dip_threshold = np.nanpercentile(normalized_flux, 1)
            dip_mask = np.isfinite(normalized_flux) & (normalized_flux <= dip_threshold)
            ax.plot(time, normalized_flux, color="#1f77b4", lw=1, label="Normalized flux")
            ax.scatter(
                time[dip_mask],
                normalized_flux[dip_mask],
                color="#d62728",
                s=18,
                label="Lowest 1% flux points",
                zorder=3,
            )
            ax.axhline(1.0, color="#2ca02c", linestyle="--", lw=1, alpha=0.8, label="Baseline")
            if flux_err is not None:
                scaled_err = flux_err / baseline if baseline not in (0, np.nan) else flux_err
                ax.fill_between(time, normalized_flux - scaled_err, normalized_flux + scaled_err, color="#1f77b4", alpha=0.15)
            ax.set_xlabel("Time")
            ax.set_ylabel("Normalized flux")
            ax.set_title(f"Light curve preview: {preview.source}")
        elif preview.dataframe is not None:
            dataframe = preview.dataframe
            time_col = self._pick_column(dataframe.columns, ["time", "bjd", "jd", "t"])
            flux_col = self._pick_column(dataframe.columns, ["flux", "pdcsap_flux", "sap_flux", "intensity", "brightness"])
            if time_col is None or flux_col is None:
                raise ValueError("Could not find time/flux columns to plot")
            time = pd.to_numeric(dataframe[time_col], errors="coerce")
            flux = pd.to_numeric(dataframe[flux_col], errors="coerce")
            ax.plot(time, flux, color="#1f77b4", lw=1, label="Flux")
            finite_flux = flux[np.isfinite(flux)]
            if len(finite_flux):
                baseline = float(np.nanmedian(finite_flux))
                ax.axhline(baseline, color="#2ca02c", linestyle="--", lw=1, alpha=0.8, label="Median baseline")
            ax.set_xlabel(time_col)
            ax.set_ylabel(flux_col)
            ax.set_title(f"Tabular preview: {preview.source}")
        else:
            raise ValueError("Preview has neither dataframe nor lightcurve data")

        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        if output_path is None and not show:
            plt.show()
            return None

        if show:
            plt.show()

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=160, bbox_inches="tight")
        else:
            output = None
        plt.close(fig)
        return output

    def format_summary(self, preview: DatasetPreview) -> str:
        lines = [f"Source: {preview.source}"]
        for key, value in preview.summary.items():
            lines.append(f"{key}: {value}")
        if preview.notes:
            lines.append("Notes:")
            lines.extend([f"- {note}" for note in preview.notes])
        return "\n".join(lines)

    def _summarize_dataframe(self, dataframe: pd.DataFrame, source: str) -> dict[str, Any]:
        numeric = dataframe.select_dtypes(include=[np.number])
        missing_counts = dataframe.isna().sum().sort_values(ascending=False)
        return {
            "source_type": "table",
            "source_name": source,
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "column_names": list(dataframe.columns),
            "numeric_columns": list(numeric.columns),
            "missing_values_top": missing_counts.head(10).to_dict(),
            "preview_rows": dataframe.head(5).to_dict(orient="records"),
        }

    def _summarize_lightcurve(self, lightcurve: LightCurve, source: str) -> dict[str, Any]:
        time = np.asarray(lightcurve.time.value if hasattr(lightcurve.time, "value") else lightcurve.time)
        flux = np.asarray(lightcurve.flux.value if hasattr(lightcurve.flux, "value") else lightcurve.flux)
        finite_flux = flux[np.isfinite(flux)]
        cadence = None
        if len(time) > 2:
            diffs = np.diff(time)
            finite_diffs = diffs[np.isfinite(diffs)]
            cadence = float(np.nanmedian(finite_diffs)) if len(finite_diffs) else None
        return {
            "source_type": "lightcurve",
            "source_name": source,
            "cadences": int(len(time)),
            "finite_flux_points": int(np.isfinite(flux).sum()),
            "time_start": float(np.nanmin(time)) if len(time) else None,
            "time_end": float(np.nanmax(time)) if len(time) else None,
            "median_cadence": cadence,
            "flux_min": float(np.nanmin(finite_flux)) if len(finite_flux) else None,
            "flux_median": float(np.nanmedian(finite_flux)) if len(finite_flux) else None,
            "flux_max": float(np.nanmax(finite_flux)) if len(finite_flux) else None,
        }

    def _read_lightcurve_file(self, file_path: Path) -> LightCurve:
        lc = LightCurve.read(file_path)
        if isinstance(lc, LightCurve):
            return lc
        if hasattr(lc, "stitch"):
            return lc.stitch()
        if isinstance(lc, Table):
            return LightCurve.from_table(lc)
        raise RuntimeError(f"Could not convert {file_path} to a LightCurve")

    @staticmethod
    def _pick_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
        lower_map = {column.lower(): column for column in columns}
        for candidate in candidates:
            if candidate in lower_map:
                return lower_map[candidate]
        return None
