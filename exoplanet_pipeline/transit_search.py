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
from matplotlib.figure import Figure
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


@dataclass
class TransitAssessment:
    """Result of vetting a BLS transit candidate."""

    source: str
    candidate: TransitCandidate | None
    confidence_percent: float
    is_candidate: bool
    status: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
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

    def analyze_path(self, path: str | Path) -> TransitAssessment:
        preview = self.inspector.load_local(path)
        lightcurve = self._preview_to_lightcurve(preview)
        return self.analyze_lightcurve(lightcurve, preview.source)

    def analyze_remote(self) -> TransitAssessment:
        lightcurve, source = self.load_target()
        return self.analyze_lightcurve(lightcurve, source)

    def analyze_lightcurve(self, lightcurve: LightCurve, source: str) -> TransitAssessment:
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

        best_period, best_transit_time, best_power, stats, depth, snr, alias_note = self._refine_period_alias(
            bls=bls,
            best_period=best_period,
            best_duration=best_duration,
            best_transit_time=best_transit_time,
            best_power=best_power,
        )
        confidence_percent, is_candidate, status, reason, metrics = self._vet_candidate(
            periodogram, candidate_index, stats, best_power, depth, snr
        )

        notes = [
            "BLS search evaluated multiple trial durations and selected the strongest periodogram peak.",
        ]
        if self.use_all_sectors:
            notes.append("All available TESS sectors were requested to extend the time baseline for periodic dip detection.")
        if alias_note is not None:
            notes.append(alias_note)
        candidate = TransitCandidate(
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
        if not is_candidate:
            return TransitAssessment(
                source=source,
                candidate=None,
                confidence_percent=confidence_percent,
                is_candidate=False,
                status=status,
                reason=reason,
                metrics=metrics,
                notes=notes + [reason],
            )
        candidate.notes.extend([f"Estimated exoplanet likelihood: {confidence_percent:.1f}%"])
        return TransitAssessment(
            source=source,
            candidate=candidate,
            confidence_percent=confidence_percent,
            is_candidate=True,
            status=status,
            reason=reason,
            metrics=metrics,
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

    def plot_candidate(self, assessment: TransitAssessment, output_path: str | Path | None = None, show: bool = False) -> Path | None:
        fig = self.build_result_figure(assessment)

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

    def build_result_figure(self, assessment: TransitAssessment, source_lightcurve: LightCurve | None = None) -> Figure:
        if assessment.candidate is None:
            return self._build_no_candidate_figure(assessment, source_lightcurve=source_lightcurve)
        return self.build_candidate_figure(assessment.candidate, source_lightcurve=source_lightcurve)

    def build_candidate_figure(self, candidate: TransitCandidate, source_lightcurve: LightCurve | None = None) -> Figure:
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)

        source_lc = source_lightcurve if source_lightcurve is not None else self._load_source_lightcurve(candidate.source)
        prepared = self.prepare_lightcurve(source_lc)
        time = self._values(prepared.time)
        flux = self._values(prepared.flux)

        self._plot_time_series(axes[0], time, flux, candidate)
        self._plot_periodogram(axes[1], candidate)
        self._plot_phase_folded(axes[2], time, flux, candidate)

        fig.suptitle(f"Transit candidate search: {candidate.source}", fontsize=14)
        return fig

    def _build_no_candidate_figure(self, assessment: TransitAssessment, source_lightcurve: LightCurve | None = None) -> Figure:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        source_lc = source_lightcurve if source_lightcurve is not None else self._load_source_lightcurve(assessment.source)
        prepared = self.prepare_lightcurve(source_lc)
        time = self._values(prepared.time)
        flux = self._values(prepared.flux)

        axes[0].plot(time, flux, color="#1f77b4", lw=1)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Normalized flux")
        axes[0].set_title("Light curve")
        axes[0].grid(True, alpha=0.25)

        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.65,
            "No transit candidates or exoplanets detected",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )
        axes[1].text(
            0.5,
            0.40,
            f"Reason: {assessment.reason}",
            ha="center",
            va="center",
            fontsize=11,
            wrap=True,
        )
        axes[1].text(
            0.5,
            0.15,
            f"Estimated likelihood: {assessment.confidence_percent:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.suptitle(f"Transit search: {assessment.source}", fontsize=14)
        return fig

    def summarize(self, assessment: TransitAssessment) -> str:
        if assessment.is_candidate:
            likelihood_line = f"Estimated exoplanet likelihood: {assessment.confidence_percent:.1f}%"
        else:
            likelihood_line = "Estimated exoplanet likelihood: negligible"
        lines = [f"Source: {assessment.source}", likelihood_line]
        lines.append(f"Status: {assessment.status}")
        if assessment.candidate is None:
            lines.append("Result: No transit candidates or exoplanets detected.")
            lines.append(f"Reason: {assessment.reason}")
        else:
            candidate = assessment.candidate
            lines.extend(
                [
                    f"Best period: {candidate.period:.6f} days",
                    f"Best duration: {candidate.duration:.6f} days",
                    f"Transit time: {candidate.transit_time:.6f} days",
                    f"Estimated depth: {candidate.depth}",
                    f"BLS power: {candidate.power:.4f}",
                    f"Signal significance / SNR: {candidate.snr}",
                ]
            )
            if candidate.stats:
                lines.append("Stats:")
                for key, value in candidate.stats.items():
                    if isinstance(value, np.ndarray):
                        continue
                    lines.append(f"- {key}: {value}")
        if assessment.metrics:
            lines.append("Vet metrics:")
            for key, value in assessment.metrics.items():
                lines.append(f"- {key}: {value}")
        if assessment.notes:
            lines.append("Notes:")
            lines.extend([f"- {note}" for note in assessment.notes])
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

    def _vet_candidate(
        self,
        periodogram: Any,
        index: int,
        stats: dict[str, Any],
        best_power: float,
        depth: float | None,
        snr: float | None,
    ) -> tuple[float, bool, str, str, dict[str, Any]]:
        power_values = np.asarray(periodogram.power, dtype=float)
        finite_power = power_values[np.isfinite(power_values)]
        if len(finite_power) == 0:
            return 0.0, False, "Rejected", "No finite BLS power values were available.", {}

        baseline = float(np.nanmedian(finite_power))
        mad = float(np.nanmedian(np.abs(finite_power - baseline)))
        spread = 1.4826 * mad if mad > 0 else float(np.nanstd(finite_power))
        spread = spread if np.isfinite(spread) and spread > 0 else 1.0
        prominence = (best_power - baseline) / spread
        ratio = best_power / (abs(baseline) + 1e-9)
        depth_snr = float(snr) if snr is not None and np.isfinite(snr) else 0.0

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        score = (
            0.5 * sigmoid((depth_snr - 6.0) / 2.0)
            + 0.3 * sigmoid((prominence - 5.0) / 1.5)
            + 0.2 * sigmoid((ratio - 1.5) / 0.7)
        )
        raw_confidence_percent = float(np.clip(score * 100.0, 0.0, 99.9))

        metrics = {
            "depth_snr": depth_snr,
            "peak_prominence_sigma": round(float(prominence), 3),
            "peak_to_background_ratio": round(float(ratio), 3),
            "baseline_power": round(float(baseline), 3),
            "power_spread": round(float(spread), 3),
            "raw_confidence_percent": round(raw_confidence_percent, 1),
        }

        if depth_snr < 6.0 or prominence < 5.0 or raw_confidence_percent < 35.0:
            reason = "The strongest BLS peak is too weak above the noise floor to count as a convincing transit candidate."
            return 0.0, False, "No convincing candidate", reason, metrics

        confidence_percent = raw_confidence_percent

        if confidence_percent >= 80.0:
            status = "Strong candidate"
        elif confidence_percent >= 55.0:
            status = "Moderate candidate"
        else:
            status = "Weak candidate"

        if depth is None:
            metrics["estimated_depth"] = "unavailable"
        return confidence_percent, True, status, "The BLS peak is sufficiently above the noise floor to treat as a transit candidate.", metrics

    def _refine_period_alias(
        self,
        bls: BoxLeastSquares,
        best_period: float,
        best_duration: float,
        best_transit_time: float,
        best_power: float,
    ) -> tuple[float, float, float, dict[str, Any], float | None, float | None, str | None]:
        alias_candidates: list[float] = [best_period]
        half_period = best_period / 2.0
        double_period = best_period * 2.0
        if self.min_period <= half_period <= self.max_period:
            alias_candidates.append(half_period)
        if self.min_period <= double_period <= self.max_period:
            alias_candidates.append(double_period)

        evaluated: list[dict[str, Any]] = []
        for period in alias_candidates:
            alias_result = bls.power(np.asarray([period]), best_duration, objective="snr")
            if len(alias_result.power) == 0:
                continue
            alias_power = float(alias_result.power[0])
            alias_transit_time = float(alias_result.transit_time[0])
            alias_stats = bls.compute_stats(period, best_duration, alias_transit_time)
            alias_depth = self._extract_depth(alias_result, 0, alias_stats)
            alias_snr = self._extract_snr(alias_stats, alias_power)
            depth_odd = alias_stats.get("depth_odd")
            depth_even = alias_stats.get("depth_even")
            odd_even_gap = np.inf
            if depth_odd is not None and depth_even is not None:
                try:
                    odd_even_gap = abs(float(depth_odd[0]) - float(depth_even[0]))
                except Exception:
                    odd_even_gap = np.inf
            evaluated.append(
                {
                    "period": period,
                    "power": alias_power,
                    "transit_time": alias_transit_time,
                    "stats": alias_stats,
                    "depth": alias_depth,
                    "snr": alias_snr,
                    "odd_even_gap": odd_even_gap,
                }
            )

        if not evaluated:
            return best_period, best_transit_time, best_power, {}, None, None, None

        def candidate_score(item: dict[str, Any]) -> float:
            power_ratio = item["power"] / (best_power + 1e-12)
            gap = item["odd_even_gap"]
            consistency = 1.0 / (1.0 + gap) if np.isfinite(gap) else 0.5
            shorter_bias = 0.03 if item["period"] < best_period else 0.0
            return 0.72 * power_ratio + 0.25 * consistency + shorter_bias

        ranked = sorted(evaluated, key=lambda item: (candidate_score(item), -item["period"]), reverse=True)
        chosen = ranked[0]
        chosen_score = candidate_score(chosen)
        best_score = candidate_score(next(item for item in evaluated if item["period"] == best_period))

        if chosen["period"] != best_period and chosen_score >= best_score * 0.98:
            alias_note = f"Alias check refined the period from {best_period:.6f} to {chosen['period']:.6f} days."
            return (
                float(chosen["period"]),
                float(chosen["transit_time"]),
                float(chosen["power"]),
                chosen["stats"],
                chosen["depth"],
                chosen["snr"],
                alias_note,
            )

        original = next(item for item in evaluated if item["period"] == best_period)
        return (
            best_period,
            best_transit_time,
            best_power,
            original["stats"],
            original["depth"],
            original["snr"],
            None,
        )

