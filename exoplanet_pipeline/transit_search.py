from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from lightkurve import LightCurve, LightCurveCollection, search_lightcurve
from lightkurve.utils import LightkurveError

from .dataset_inspector import DatasetInspector, DatasetPreview


@dataclass
class TransitCandidate:
    """Best periodic dip candidate found in a light curve."""

    source: str
    period: float
    duration: float
    transit_time: float
    depth: float | None
    power: float
    snr: float | None
    stats: dict[str, Any] = field(default_factory=dict)
    periodogram_periods: np.ndarray | None = None
    periodogram_power: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


class TransitSearcher:
    """Search for transit-like periodic dips using a BLS periodogram."""

    def __init__(
        self,
        target_id: str | None = None,
        sector: int | None = None,
        use_all_sectors: bool = False,
        min_period: float = 0.5,
        max_period: float = 20.0,
        min_duration: float = 0.05,
        max_duration: float = 0.3,
        duration_steps: int = 8,
        period_samples: int = 5000,
    ):
        self.target_id = target_id
        self.sector = sector
        self.use_all_sectors = use_all_sectors
        self.min_period = min_period
        self.max_period = max_period
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.duration_steps = duration_steps
        self.period_samples = period_samples
        self.inspector = DatasetInspector(target_id=target_id, sector=sector)

    def load_target(self) -> tuple[LightCurve, str]:
        if not self.target_id:
            raise ValueError("target_id is required to load a remote TESS target")

        query_candidates = self._build_query_candidates(self.target_id)
        search = None
        query_target = query_candidates[0]

        for candidate in query_candidates:
            query_target = candidate
            try:
                search = (
                    search_lightcurve(candidate, mission="TESS")
                    if self.use_all_sectors
                    else search_lightcurve(candidate, mission="TESS", sector=self.sector)
                )
            except Exception:
                search = None

            if search is not None and len(search) > 0:
                break

            if self.sector is not None and not self.use_all_sectors:
                try:
                    search = search_lightcurve(candidate, mission="TESS")
                except Exception:
                    search = None
                if search is not None and len(search) > 0:
                    break

        if search is None or len(search) == 0:
            raise RuntimeError(
                f"No TESS light curves found for {query_target}. "
                "This usually means the ID is not a valid TIC target with public TESS data. "
                "Try another TIC ID, use --all-sectors only with a known TESS target, or inspect a local FITS/CSV file instead."
            )

        downloaded = self._download_with_retry(search)
        lightcurve = self._stitch_download(downloaded)
        return lightcurve, query_target

    @staticmethod
    def _build_query_candidates(target_id: str) -> list[str]:
        cleaned = str(target_id).strip()
        if cleaned.upper().startswith("TIC"):
            numeric = cleaned[3:].strip()
            return [cleaned, numeric] if numeric else [cleaned]
        return [f"TIC {cleaned}", cleaned]

    def analyze_path(self, path: str | Path) -> TransitCandidate:
        preview = self.inspector.load_local(path)
        lightcurve = self._preview_to_lightcurve(preview)
        return self.analyze_lightcurve(lightcurve, preview.source)

    def analyze_remote(self) -> TransitCandidate:
        lightcurve, source = self.load_target()
        return self.analyze_lightcurve(lightcurve, source)

    def analyze_lightcurve(self, lightcurve: LightCurve, source: str) -> TransitCandidate:
        prepared = self.prepare_lightcurve(lightcurve)
        time = self._values(prepared.time)
        flux = self._values(prepared.flux)
        flux_err = self._values(prepared.flux_err) if getattr(prepared, "flux_err", None) is not None else None
        if flux_err is not None and np.isfinite(flux_err).sum() == 0:
            flux_err = None

        duration_grid = np.linspace(self.min_duration, self.max_duration, self.duration_steps)
        period_grid = np.linspace(self.min_period, self.max_period, self.period_samples)
        best_result = None
        best_duration = None
        best_duration_score = -np.inf

        for duration in duration_grid:
            bls = BoxLeastSquares(time, flux, dy=flux_err)
            periodogram = bls.power(period_grid, duration, objective="snr")
            if len(periodogram.power) == 0:
                continue
            candidate_index = int(np.argmax(periodogram.power))
            candidate_power = float(periodogram.power[candidate_index])
            if candidate_power > best_duration_score:
                best_duration_score = candidate_power
                best_duration = float(duration)
                best_result = (bls, periodogram, candidate_index)

        if best_result is None or best_duration is None:
            raise RuntimeError("Could not evaluate any BLS candidates for the supplied light curve")

        bls, periodogram, candidate_index = best_result
        best_period = float(periodogram.period[candidate_index])
        best_transit_time = float(periodogram.transit_time[candidate_index])
        best_power = float(periodogram.power[candidate_index])

        stats = bls.compute_stats(best_period, best_duration, best_transit_time)
        depth = self._extract_depth(periodogram, candidate_index, stats)
        snr = self._extract_snr(stats, best_power)

        notes = [
            "BLS search evaluated multiple trial durations and selected the strongest periodogram peak.",
        ]
        if self.use_all_sectors:
            notes.append("All available TESS sectors were requested to extend the time baseline for periodic dip detection.")
        return TransitCandidate(
            source=source,
            period=best_period,
            duration=best_duration,
            transit_time=best_transit_time,
            depth=depth,
            power=best_power,
            snr=snr,
            stats=stats,
            periodogram_periods=np.asarray(periodogram.period),
            periodogram_power=np.asarray(periodogram.power),
            notes=notes,
        )

    def prepare_lightcurve(self, lightcurve: LightCurve) -> LightCurve:
        cleaned = lightcurve.remove_nans()
        if len(cleaned) == 0:
            raise RuntimeError("The light curve has no finite cadences after removing NaNs")

        if len(cleaned) > 20:
            cleaned = cleaned.remove_outliers(sigma=7)

        normalized = cleaned.normalize()
        window_length = self._choose_window_length(len(normalized))
        flattened = normalized.flatten(window_length=window_length, polyorder=2)
        return flattened

    def plot_candidate(self, candidate: TransitCandidate, output_path: str | Path | None = None, show: bool = False) -> Path | None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)

        source_lc = self._load_source_lightcurve(candidate.source)
        prepared = self.prepare_lightcurve(source_lc)
        time = self._values(prepared.time)
        flux = self._values(prepared.flux)

        self._plot_time_series(axes[0], time, flux, candidate)
        self._plot_periodogram(axes[1], candidate)
        self._plot_phase_folded(axes[2], time, flux, candidate)

        fig.suptitle(f"Transit candidate search: {candidate.source}", fontsize=14)

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=170, bbox_inches="tight")
        else:
            output = None

        if show:
            plt.show()

        plt.close(fig)
        return output

    def summarize(self, candidate: TransitCandidate) -> str:
        lines = [
            f"Source: {candidate.source}",
            f"Best period: {candidate.period:.6f} days",
            f"Best duration: {candidate.duration:.6f} days",
            f"Transit time: {candidate.transit_time:.6f} days",
            f"Estimated depth: {candidate.depth}",
            f"BLS power: {candidate.power:.4f}",
            f"Signal significance / SNR: {candidate.snr}",
        ]
        if candidate.stats:
            lines.append("Stats:")
            for key, value in candidate.stats.items():
                if isinstance(value, np.ndarray):
                    continue
                lines.append(f"- {key}: {value}")
        if candidate.notes:
            lines.append("Notes:")
            lines.extend([f"- {note}" for note in candidate.notes])
        return "\n".join(lines)

    def _load_source_lightcurve(self, source: str) -> LightCurve:
        if source.lower().endswith((".csv", ".txt", ".parquet", ".fits", ".fit", ".lc")):
            preview = self.inspector.load_local(source)
            return self._preview_to_lightcurve(preview)
        if self.target_id:
            lightcurve, _ = self.load_target()
            return lightcurve
        raise ValueError("Unable to infer how to load the source light curve")

    def _preview_to_lightcurve(self, preview: DatasetPreview) -> LightCurve:
        if preview.lightcurve is not None:
            return preview.lightcurve

        if preview.dataframe is None:
            raise ValueError("Preview does not contain usable light-curve data")

        dataframe = preview.dataframe.copy()
        time_col = self.inspector._pick_column(dataframe.columns, ["time", "bjd", "jd", "t"])
        flux_col = self.inspector._pick_column(dataframe.columns, ["flux", "pdcsap_flux", "sap_flux", "intensity", "brightness"])
        if time_col is None or flux_col is None:
            raise ValueError("Could not find time and flux columns in the supplied file")

        time = pd.to_numeric(dataframe[time_col], errors="coerce").to_numpy()
        flux = pd.to_numeric(dataframe[flux_col], errors="coerce").to_numpy()
        flux_err_col = self.inspector._pick_column(dataframe.columns, ["flux_err", "flux_error", "error", "err"])
        flux_err = None
        if flux_err_col is not None:
            flux_err = pd.to_numeric(dataframe[flux_err_col], errors="coerce").to_numpy()

        return LightCurve(time=time, flux=flux, flux_err=flux_err)

    def _stitch_download(self, downloaded: Any) -> LightCurve:
        if isinstance(downloaded, LightCurve):
            return downloaded
        if hasattr(downloaded, "stitch"):
            return downloaded.stitch()
        if isinstance(downloaded, list):
            if len(downloaded) == 1 and isinstance(downloaded[0], LightCurve):
                return downloaded[0]
            return LightCurveCollection(downloaded).stitch()  # type: ignore[name-defined]
        raise RuntimeError("Could not convert downloaded products into a single LightCurve")

    def _download_with_retry(self, search: Any) -> Any:
        try:
            return search.download_all() if len(search) > 1 else search.download()
        except LightkurveError as exc:
            cleaned_any = self._clear_corrupt_cache_entries(str(exc))
            if cleaned_any:
                try:
                    return search.download_all() if len(search) > 1 else search.download()
                except LightkurveError:
                    pass
            if len(search) > 1:
                for index in range(len(search)):
                    try:
                        return search[index : index + 1].download_all()
                    except LightkurveError:
                        continue
            raise

    def _clear_corrupt_cache_entries(self, message: str) -> bool:
        matches = re.findall(r"Data product (.+?) of type generic", message)
        cleaned_any = False
        for match in matches:
            file_path = Path(match)
            if file_path.exists():
                try:
                    file_path.unlink()
                    cleaned_any = True
                except OSError:
                    continue
        return cleaned_any

    def _choose_window_length(self, size: int) -> int:
        if size < 11:
            return max(3, size - 1 if size % 2 == 0 else size)
        window = min(max(51, size // 10), 501)
        if window >= size:
            window = size - 1 if size % 2 == 0 else size
        if window % 2 == 0:
            window -= 1
        return max(window, 5)

    def _plot_time_series(self, axis: Any, time: np.ndarray, flux: np.ndarray, candidate: TransitCandidate) -> None:
        axis.plot(time, flux, color="#1f77b4", lw=1, label="Detrended flux")
        axis.axhline(1.0, color="#2ca02c", linestyle="--", lw=1, alpha=0.8, label="Baseline")
        transit_centers = self._transit_centers(time, candidate.period, candidate.transit_time)
        for center in transit_centers:
            axis.axvline(center, color="#d62728", alpha=0.18, lw=1)
        axis.set_xlabel("Time")
        axis.set_ylabel("Normalized flux")
        axis.set_title("Light curve with predicted transit times")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)

    def _plot_periodogram(self, axis: Any, candidate: TransitCandidate) -> None:
        axis.plot(candidate.periodogram_periods, candidate.periodogram_power, color="#9467bd", lw=1.2)
        axis.axvline(candidate.period, color="#d62728", linestyle="--", lw=1.2, label=f"Best period = {candidate.period:.4f} d")
        axis.set_xlabel("Period (days)")
        axis.set_ylabel("BLS power")
        axis.set_title("Box Least Squares periodogram")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)

    def _plot_phase_folded(self, axis: Any, time: np.ndarray, flux: np.ndarray, candidate: TransitCandidate) -> None:
        phase = ((time - candidate.transit_time + 0.5 * candidate.period) % candidate.period) - 0.5 * candidate.period
        axis.scatter(phase, flux, s=9, color="#1f77b4", alpha=0.65, edgecolors="none", label="Folded flux")
        half_duration = candidate.duration / 2.0
        axis.axvspan(-half_duration, half_duration, color="#d62728", alpha=0.15, label="Transit window")
        axis.set_xlabel("Phase (days)")
        axis.set_ylabel("Normalized flux")
        axis.set_title("Phase-folded light curve")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)

    def _transit_centers(self, time: np.ndarray, period: float, transit_time: float) -> list[float]:
        if len(time) == 0:
            return []
        start = float(np.nanmin(time))
        end = float(np.nanmax(time))
        first_index = int(np.floor((start - transit_time) / period)) - 1
        last_index = int(np.ceil((end - transit_time) / period)) + 1
        centers = [transit_time + idx * period for idx in range(first_index, last_index + 1)]
        return [center for center in centers if start <= center <= end]

    @staticmethod
    def _values(column: Any) -> np.ndarray:
        if hasattr(column, "value"):
            return np.asarray(column.value)
        return np.asarray(column)

    @staticmethod
    def _extract_depth(periodogram: Any, index: int, stats: dict[str, Any]) -> float | None:
        depth = None
        if hasattr(periodogram, "depth"):
            depth_values = np.asarray(periodogram.depth)
            if len(depth_values) > index:
                depth = float(depth_values[index])
        if depth is None and "depth" in stats:
            try:
                depth = float(stats["depth"])
            except Exception:
                depth = None
        return depth

    @staticmethod
    def _extract_snr(stats: dict[str, Any], fallback: float) -> float | None:
        for key in ("depth_snr", "snr"):
            if key in stats:
                try:
                    return float(stats[key])
                except Exception:
                    pass
        return float(fallback)

