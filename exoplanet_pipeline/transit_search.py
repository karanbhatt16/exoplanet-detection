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
from astroquery.mast import Catalogs
from matplotlib.figure import Figure
from lightkurve import LightCurve, LightCurveCollection, search_lightcurve
from lightkurve.utils import LightkurveError
from scipy.optimize import least_squares

try:
    import batman
except Exception:  # pragma: no cover - optional dependency fallback
    batman = None

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
    stellar_radius_rsun: float | None = None
    stellar_radius_err_rsun: float | None = None
    stellar_mass_msun: float | None = None
    stellar_mass_err_msun: float | None = None
    planet_radius_rsun: float | None = None
    planet_radius_err_rsun: float | None = None
    planet_radius_rearth: float | None = None
    planet_radius_err_rearth: float | None = None
    radius_ratio: float | None = None
    radius_source: str | None = None
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
        period_power_tolerance: float = 0.97,
        stellar_radius_rsun: float | None = None,
        stellar_radius_err_rsun: float | None = None,
        stellar_mass_msun: float | None = None,
        stellar_mass_err_msun: float | None = None,
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
        self.period_power_tolerance = period_power_tolerance
        self.stellar_radius_rsun = stellar_radius_rsun
        self.stellar_radius_err_rsun = stellar_radius_err_rsun
        self.stellar_mass_msun = stellar_mass_msun
        self.stellar_mass_err_msun = stellar_mass_err_msun
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
        radius_prepared = self.prepare_radius_lightcurve(lightcurve)
        time = self._values(prepared.time)
        flux = self._values(prepared.flux)
        radius_time = self._values(radius_prepared.time)
        radius_flux = self._values(radius_prepared.flux)
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
        best_period, best_transit_time, best_power, period_note = self._choose_period_solution(periodogram)

        best_period, best_transit_time, best_power, stats, depth, snr, alias_note = self._refine_period_alias(
            bls=bls,
            best_period=best_period,
            best_duration=best_duration,
            best_transit_time=best_transit_time,
            best_power=best_power,
        )
        refined_depth, depth_err, depth_note = self._refine_transit_depth(
            radius_time, radius_flux, best_period, best_transit_time, best_duration, depth
        )
        if refined_depth is not None:
            depth = refined_depth
            stats = dict(stats)
            stats["depth"] = (depth, depth_err) if depth_err is not None else depth
            if depth_note is not None:
                stats["depth_note"] = depth_note

        confidence_percent, is_candidate, status, reason, metrics = self._vet_candidate(
            periodogram, candidate_index, stats, best_power, depth, snr
        )

        notes = [
            "BLS search evaluated multiple trial durations and selected the strongest periodogram peak.",
        ]
        if period_note is not None:
            notes.append(period_note)
        if self.use_all_sectors:
            notes.append("All available TESS sectors were requested to extend the time baseline for periodic dip detection.")
        if alias_note is not None:
            notes.append(alias_note)
        stellar_radius_rsun, stellar_radius_err_rsun, stellar_mass_msun, stellar_mass_err_msun, radius_source = self._resolve_stellar_properties(source)
        planet_radius_rsun, planet_radius_err_rsun, planet_radius_rearth, planet_radius_err_rearth, radius_ratio = (
            self._estimate_planet_radius(
                radius_time,
                radius_flux,
                depth,
                stats,
                best_period,
                best_transit_time,
                best_duration,
                stellar_radius_rsun,
                stellar_radius_err_rsun,
                stellar_mass_msun,
                stellar_mass_err_msun,
            )
        )

        candidate = TransitCandidate(
            source=source,
            period=best_period,
            duration=best_duration,
            transit_time=best_transit_time,
            depth=depth,
            power=best_power,
            snr=snr,
            stellar_radius_rsun=stellar_radius_rsun,
            stellar_radius_err_rsun=stellar_radius_err_rsun,
            stellar_mass_msun=stellar_mass_msun,
            stellar_mass_err_msun=stellar_mass_err_msun,
            planet_radius_rsun=planet_radius_rsun,
            planet_radius_err_rsun=planet_radius_err_rsun,
            planet_radius_rearth=planet_radius_rearth,
            planet_radius_err_rearth=planet_radius_err_rearth,
            radius_ratio=radius_ratio,
            radius_source=radius_source,
            stats=stats,
            periodogram_periods=np.asarray(periodogram.period),
            periodogram_power=np.asarray(periodogram.power),
            notes=notes,
        )
        if not is_candidate:
            candidate.notes.extend(
                [
                    "This is the closest BLS candidate, but the signal is weak and should be treated cautiously.",
                    reason,
                ]
            )
        else:
            candidate.notes.extend([f"Estimated exoplanet likelihood: {confidence_percent:.1f}%"])
        return TransitAssessment(
            source=source,
            candidate=candidate,
            confidence_percent=confidence_percent,
            is_candidate=is_candidate,
            status=status,
            reason=reason,
            metrics=self._augment_radius_metrics(metrics, stellar_radius_rsun, stellar_radius_err_rsun, planet_radius_rearth, planet_radius_err_rearth),
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

    def prepare_radius_lightcurve(self, lightcurve: LightCurve) -> LightCurve:
        cleaned = lightcurve.remove_nans()
        if len(cleaned) == 0:
            raise RuntimeError("The light curve has no finite cadences after removing NaNs")

        if len(cleaned) > 20:
            cleaned = cleaned.remove_outliers(sigma=7)

        return cleaned.normalize()

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
        return self.build_dashboard_figure(assessment, source_lightcurve=source_lightcurve)

    def build_candidate_figure(self, candidate: TransitCandidate, source_lightcurve: LightCurve | None = None) -> Figure:
        fig = self.build_dashboard_figure(
            TransitAssessment(
                source=candidate.source,
                candidate=candidate,
                confidence_percent=0.0,
                is_candidate=True,
                status="Candidate",
                reason="",
                metrics={},
                notes=[],
            ),
            source_lightcurve=source_lightcurve,
        )
        return fig

    def build_dashboard_figure(self, assessment: TransitAssessment, source_lightcurve: LightCurve | None = None) -> Figure:
        fig = plt.figure(figsize=(18, 10), facecolor="#081018")
        fig.subplots_adjust(left=0.03, right=0.985, top=0.92, bottom=0.08, wspace=0.18, hspace=0.22)
        gs = fig.add_gridspec(2, 3)

        raw_ax = fig.add_subplot(gs[0, 0])
        pipeline_ax = fig.add_subplot(gs[0, 1])
        summary_ax = fig.add_subplot(gs[0, 2])
        denoised_ax = fig.add_subplot(gs[1, 0])
        score_ax = fig.add_subplot(gs[1, 1])
        phase_ax = fig.add_subplot(gs[1, 2])

        axes = [raw_ax, pipeline_ax, summary_ax, denoised_ax, score_ax, phase_ax]
        for axis in axes:
            axis.set_facecolor("#0d1520")
            axis.tick_params(colors="#dbe7f3", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#2f4158")

        source_lc = source_lightcurve if source_lightcurve is not None else self._load_source_lightcurve(assessment.source)
        raw_lc = self.prepare_radius_lightcurve(source_lc)
        denoised_lc = self.prepare_lightcurve(source_lc)

        raw_time = self._values(raw_lc.time)
        raw_flux = self._values(raw_lc.flux)
        den_time = self._values(denoised_lc.time)
        den_flux = self._values(denoised_lc.flux)

        candidate = assessment.candidate
        if candidate is None:
            candidate = TransitCandidate(
                source=assessment.source,
                period=1.0,
                duration=1.0 / 24.0,
                transit_time=float(raw_time[0]) if len(raw_time) else 0.0,
                depth=None,
                power=0.0,
                snr=None,
            )

        self._plot_raw_lightcurve_card(raw_ax, raw_time, raw_flux, candidate, assessment)
        self._plot_pipeline_card(pipeline_ax, assessment)
        self._plot_summary_card(summary_ax, assessment)
        self._plot_denoised_card(denoised_ax, den_time, den_flux, candidate)
        self._plot_score_card(score_ax, candidate)
        self._plot_phase_folded_card(phase_ax, den_time, den_flux, candidate)

        fig.text(
            0.5,
            0.015,
            "Exoplanet transit search dashboard",
            ha="center",
            va="bottom",
            color="white",
            fontsize=16,
            fontweight="bold",
        )
        return fig

    def _status_color(self, assessment: TransitAssessment) -> str:
        if assessment.candidate is None or assessment.confidence_percent < 35.0:
            return "#ff5c77"
        if assessment.confidence_percent < 80.0:
            return "#f5c542"
        return "#4ade80"

    def _plot_raw_lightcurve_card(
        self,
        axis: Any,
        time: np.ndarray,
        flux: np.ndarray,
        candidate: TransitCandidate,
        assessment: TransitAssessment,
    ) -> None:
        axis.plot(time, flux, color="#e8eef7", lw=0.8, alpha=0.9)
        axis.scatter(time[:: max(len(time) // 250, 1)], flux[:: max(len(flux) // 250, 1)], s=4, color="#d7deea", alpha=0.5)
        transit_centers = self._transit_centers(time, candidate.period, candidate.transit_time)
        for center in transit_centers:
            axis.axvline(center, color="#f59e0b", alpha=0.15, lw=1)
        axis.set_title("1. Noisy Light Curve", loc="left", color="#9ad1ff", fontsize=12, fontweight="bold")
        axis.set_xlabel("Time (days)", color="#dbe7f3")
        axis.set_ylabel("Relative Flux", color="#dbe7f3")
        axis.grid(True, alpha=0.18, color="#2f4158")
        axis.text(
            0.01,
            0.94,
            "Raw flux measurements over time",
            transform=axis.transAxes,
            color="#aab8c7",
            fontsize=9,
            va="top",
        )

    def _plot_pipeline_card(self, axis: Any, assessment: TransitAssessment) -> None:
        axis.set_axis_off()
        axis.set_title("2. AI Pipeline", loc="left", color="#9ad1ff", fontsize=12, fontweight="bold")
        axis.text(0.02, 0.92, "Deep learning style workflow for transit detection", color="#aab8c7", fontsize=9, transform=axis.transAxes)

        boxes = [
            (0.03, 0.40, 0.16, 0.32, "Input\nNoisy\nLight Curve"),
            (0.24, 0.40, 0.16, 0.32, "Denoising\nFlattening"),
            (0.45, 0.40, 0.16, 0.32, "Feature\nExtraction"),
            (0.66, 0.40, 0.16, 0.32, "Transit\nDetection"),
            (0.87, 0.40, 0.10, 0.32, f"Output\n{assessment.confidence_percent:.2f}%"),
        ]
        for idx, (x, y, w, h, label) in enumerate(boxes):
            rect = plt.Rectangle((x, y), w, h, transform=axis.transAxes, facecolor="#0b1320", edgecolor="#38506a", lw=1.2)
            axis.add_patch(rect)
            axis.text(x + w / 2, y + h / 2, label, color="white", ha="center", va="center", fontsize=9, transform=axis.transAxes)
            if idx < len(boxes) - 1:
                axis.annotate(
                    "",
                    xy=(x + w + 0.01, y + h / 2),
                    xytext=(x + w + 0.055, y + h / 2),
                    xycoords=axis.transAxes,
                    textcoords=axis.transAxes,
                    arrowprops=dict(arrowstyle="->", color="#7aa2d6", lw=1.2),
                )

        axis.text(
            0.5,
            0.12,
            "Transit Detected" if assessment.confidence_percent >= 35.0 else "Weak Candidate",
            color=self._status_color(assessment),
            fontsize=14,
            fontweight="bold",
            ha="center",
            transform=axis.transAxes,
        )

    def _plot_summary_card(self, axis: Any, assessment: TransitAssessment) -> None:
        axis.set_axis_off()
        axis.set_title("3. Detection Result", loc="left", color="#9ad1ff", fontsize=12, fontweight="bold")
        candidate = assessment.candidate
        status_color = self._status_color(assessment)
        status_text = assessment.status if candidate is not None else "No Candidate"
        lines = [
            f"Status: {status_text}",
            f"Confidence: {assessment.confidence_percent:.2f}%",
            f"Period: {candidate.period:.4f} days" if candidate else "Period: n/a",
            f"Depth: {candidate.depth:.4f}" if candidate and candidate.depth is not None else "Depth: n/a",
            f"Duration: {candidate.duration * 24.0:.2f} hours" if candidate else "Duration: n/a",
            f"Radius: {candidate.planet_radius_rearth:.2f} R_earth" if candidate and candidate.planet_radius_rearth is not None else "Radius: n/a",
        ]
        axis.text(
            0.05,
            0.86,
            lines[0],
            color=status_color,
            fontsize=14,
            fontweight="bold",
            transform=axis.transAxes,
        )
        for index, line in enumerate(lines[1:], start=1):
            axis.text(0.06, 0.86 - index * 0.13, f"• {line}", color="#dbe7f3", fontsize=10, transform=axis.transAxes)

    def _plot_denoised_card(self, axis: Any, time: np.ndarray, flux: np.ndarray, candidate: TransitCandidate) -> None:
        axis.plot(time, flux, color="#5eead4", lw=0.9)
        axis.set_title("4. Model Denoised Light Curve", loc="left", color="#9ad1ff", fontsize=12, fontweight="bold")
        axis.set_xlabel("Time (days)", color="#dbe7f3")
        axis.set_ylabel("Relative Flux", color="#dbe7f3")
        axis.grid(True, alpha=0.18, color="#2f4158")
        transit_centers = self._transit_centers(time, candidate.period, candidate.transit_time)
        for center in transit_centers:
            axis.axvline(center, color="#22d3ee", alpha=0.12, lw=1)

    def _plot_score_card(self, axis: Any, candidate: TransitCandidate) -> None:
        axis.set_title("5. Transit Probability / Score", loc="left", color="#9ad1ff", fontsize=12, fontweight="bold")
        periods = candidate.periodogram_periods if candidate.periodogram_periods is not None else np.asarray([])
        powers = candidate.periodogram_power if candidate.periodogram_power is not None else np.asarray([])
        if len(periods) and len(powers):
            finite = np.isfinite(periods) & np.isfinite(powers)
            periods = periods[finite]
            powers = powers[finite]
        if len(periods) == 0 or len(powers) == 0:
            axis.text(0.5, 0.5, "No periodogram available", color="#dbe7f3", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            return
        min_power = float(np.nanmin(powers))
        max_power = float(np.nanmax(powers))
        norm = (powers - min_power) / (max(max_power - min_power, 1e-9))
        axis.plot(periods, norm, color="#a855f7", lw=1.1, label="Normalized BLS score")
        axis.axvline(candidate.period, color="#f59e0b", lw=1.1, ls="--", label="Selected period")
        axis.axhline(0.5, color="#ff6b6b", lw=1.0, ls=":", label="Threshold")
        axis.set_xlabel("Period (days)", color="#dbe7f3")
        axis.set_ylabel("Score", color="#dbe7f3")
        axis.grid(True, alpha=0.18, color="#2f4158")
        axis.legend(loc="upper right", fontsize=8, facecolor="#0d1520", edgecolor="#2f4158")

    def _plot_phase_folded_card(self, axis: Any, time: np.ndarray, flux: np.ndarray, candidate: TransitCandidate) -> None:
        phase = ((time - candidate.transit_time + 0.5 * candidate.period) % candidate.period) - 0.5 * candidate.period
        axis.scatter(phase, flux, s=8, color="#67e8f9", alpha=0.7, edgecolors="none", label="Data")
        half_duration = candidate.duration / 2.0
        axis.axvspan(-half_duration, half_duration, color="#f59e0b", alpha=0.12, label="Transit Window")
        axis.set_title("6. Phase-folded Light Curve", loc="left", color="#9ad1ff", fontsize=12, fontweight="bold")
        axis.set_xlabel("Phase (days)", color="#dbe7f3")
        axis.set_ylabel("Relative Flux", color="#dbe7f3")
        axis.grid(True, alpha=0.18, color="#2f4158")
        axis.legend(loc="best", fontsize=8, facecolor="#0d1520", edgecolor="#2f4158")

    def summarize(self, assessment: TransitAssessment) -> str:
        lines = [f"Source: {assessment.source}", f"Estimated exoplanet likelihood: {assessment.confidence_percent:.1f}%"]
        lines.append(f"Status: {assessment.status}")
        candidate = assessment.candidate
        if candidate is None:
            lines.append("Result: No candidate object was produced.")
            lines.append(f"Reason: {assessment.reason}")
            return "\n".join(lines)

        lines.extend(
            [
                f"Best period: {candidate.period:.6f} days",
                f"Best duration: {candidate.duration * 24.0:.3f} hours",
                f"Transit time: {candidate.transit_time:.6f} days",
                f"Estimated depth: {candidate.depth}",
                f"BLS power: {candidate.power:.4f}",
                f"Signal significance / SNR: {candidate.snr}",
            ]
        )
        if candidate.radius_ratio is not None:
            lines.append(f"Radius ratio (Rp/Rs): {candidate.radius_ratio:.5f}")
        if candidate.stellar_radius_rsun is not None:
            if candidate.stellar_radius_err_rsun is not None:
                lines.append(
                    f"Stellar radius: {candidate.stellar_radius_rsun:.4f} +/- {candidate.stellar_radius_err_rsun:.4f} R_sun"
                )
            else:
                lines.append(f"Stellar radius: {candidate.stellar_radius_rsun:.4f} R_sun")
        if candidate.planet_radius_rearth is not None:
            if candidate.planet_radius_err_rearth is not None:
                lines.append(
                    f"Estimated planet radius: {candidate.planet_radius_rearth:.3f} +/- {candidate.planet_radius_err_rearth:.3f} R_earth"
                )
            else:
                lines.append(f"Estimated planet radius: {candidate.planet_radius_rearth:.3f} R_earth")
        elif candidate.planet_radius_rsun is not None:
            lines.append(f"Estimated planet radius: {candidate.planet_radius_rsun:.5f} R_sun")
        if candidate.radius_source is not None:
            lines.append(f"Radius source: {candidate.radius_source}")
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

    def _resolve_stellar_properties(
        self, source: str
    ) -> tuple[float | None, float | None, float | None, float | None, str | None]:
        if self.stellar_radius_rsun is not None and self.stellar_radius_rsun > 0:
            return (
                self.stellar_radius_rsun,
                self.stellar_radius_err_rsun,
                self.stellar_mass_msun,
                self.stellar_mass_err_msun,
                "manual input",
            )

        tic_id = self._extract_tic_id(source)
        if tic_id is None:
            return None, None, None, None, None

        try:
            table = Catalogs.query_object(f"TIC {tic_id}", catalog="Tic")
        except Exception:
            return None, None, None, None, None

        for row in table:
            try:
                if int(row["ID"]) != int(tic_id):
                    continue
            except Exception:
                continue

            stellar_radius = self._safe_float(row.get("rad"))
            stellar_radius_err = self._safe_float(row.get("e_rad"))
            stellar_mass = self._safe_float(row.get("mass"))
            stellar_mass_err = self._safe_float(row.get("e_mass"))
            if stellar_radius is not None and stellar_radius > 0:
                return stellar_radius, stellar_radius_err, stellar_mass, stellar_mass_err, "TIC catalog"

        return None, None, None, None, None

    @staticmethod
    def _extract_tic_id(source: str) -> int | None:
        match = re.search(r"(?:TIC\s*)?(\d{6,})", str(source), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if np.ma.is_masked(value):
                return None
        except Exception:
            pass
        try:
            value_str = str(value).strip()
            if not value_str or value_str == "--":
                return None
            converted = float(value)
            if not np.isfinite(converted):
                return None
            return converted
        except Exception:
            return None

    def _estimate_planet_radius(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        depth: float | None,
        stats: dict[str, Any],
        period: float,
        transit_time: float,
        duration: float,
        stellar_radius_rsun: float | None,
        stellar_radius_err_rsun: float | None,
        stellar_mass_msun: float | None,
        stellar_mass_err_msun: float | None,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        if depth is None or depth <= 0:
            return None, None, None, None, None
        if stellar_radius_rsun is None or stellar_radius_rsun <= 0:
            return None, None, None, None, None

        radius_ratio = float(np.sqrt(depth))
        fitted_ratio = self._fit_radius_ratio_with_batman(
            time=time,
            flux=flux,
            period=period,
            transit_time=transit_time,
            duration=duration,
            stellar_radius_rsun=stellar_radius_rsun,
            stellar_mass_msun=stellar_mass_msun,
            initial_radius_ratio=radius_ratio,
        )
        if fitted_ratio is not None:
            radius_ratio = fitted_ratio

        planet_radius_rsun = stellar_radius_rsun * radius_ratio

        planet_radius_err_rsun = None
        depth_err = None
        depth_info = stats.get("depth")
        if isinstance(depth_info, (tuple, list)) and len(depth_info) >= 2:
            try:
                depth_err = float(depth_info[1])
            except Exception:
                depth_err = None

        relative_terms: list[float] = []
        if depth_err is not None and depth_err > 0:
            relative_terms.append(0.5 * (depth_err / depth))
        if stellar_radius_err_rsun is not None and stellar_radius_err_rsun > 0:
            relative_terms.append(stellar_radius_err_rsun / stellar_radius_rsun)
        if relative_terms:
            planet_radius_err_rsun = planet_radius_rsun * float(np.sqrt(np.sum(np.square(relative_terms))))

        planet_radius_rearth = float((planet_radius_rsun * u.R_sun).to(u.R_earth).value)
        planet_radius_err_rearth = None
        if planet_radius_err_rsun is not None:
            planet_radius_err_rearth = float((planet_radius_err_rsun * u.R_sun).to(u.R_earth).value)

        return planet_radius_rsun, planet_radius_err_rsun, planet_radius_rearth, planet_radius_err_rearth, radius_ratio

    def _fit_radius_ratio_with_batman(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        transit_time: float,
        duration: float,
        stellar_radius_rsun: float,
        stellar_mass_msun: float | None,
        initial_radius_ratio: float,
    ) -> float | None:
        if batman is None or stellar_mass_msun is None or stellar_mass_msun <= 0:
            return None

        finite = np.isfinite(time) & np.isfinite(flux)
        if not np.any(finite):
            return None

        phase = ((time - transit_time + 0.5 * period) % period) - 0.5 * period
        window = np.abs(phase) <= max(duration * 2.5, duration + 0.1)
        mask = finite & window
        if int(np.sum(mask)) < 50:
            mask = finite
        if int(np.sum(mask)) < 30:
            return None

        fit_time = time[mask]
        fit_flux = flux[mask]

        from astropy.constants import G, M_sun, R_sun

        period_seconds = period * 86400.0
        a_m = ((G.value * stellar_mass_msun * M_sun.value * period_seconds**2) / (4.0 * np.pi**2)) ** (1.0 / 3.0)
        a_rs = a_m / (stellar_radius_rsun * R_sun.value)
        if not np.isfinite(a_rs) or a_rs <= 1.0:
            return None

        params = batman.TransitParams()
        params.t0 = transit_time
        params.per = period
        params.rp = float(np.clip(initial_radius_ratio, 0.005, 0.4))
        params.a = float(a_rs)
        params.inc = 89.0
        params.ecc = 0.0
        params.w = 90.0
        params.u = [0.3, 0.2]
        params.limb_dark = "quadratic"
        model = batman.TransitModel(params, fit_time)

        def residuals(values: np.ndarray) -> np.ndarray:
            rp, inc, offset = values
            params.rp = float(np.clip(rp, 0.001, 0.8))
            params.inc = float(np.clip(inc, 75.0, 90.0))
            return model.light_curve(params) + offset - fit_flux

        initial = np.asarray([params.rp, params.inc, 0.0], dtype=float)
        bounds = ([0.001, 75.0, -0.05], [0.8, 90.0, 0.05])
        try:
            result = least_squares(residuals, initial, bounds=bounds, max_nfev=200)
        except Exception:
            return None

        if not result.success:
            return None

        fitted_rp = float(result.x[0])
        if not np.isfinite(fitted_rp) or fitted_rp <= 0:
            return None
        return fitted_rp

    @staticmethod
    def _refine_transit_depth(
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        transit_time: float,
        duration: float,
        fallback_depth: float | None,
    ) -> tuple[float | None, float | None, str | None]:
        if fallback_depth is None or fallback_depth <= 0:
            return None, None, None
        if len(time) == 0 or len(flux) == 0:
            return fallback_depth, None, None

        phase = ((time - transit_time + 0.5 * period) % period) - 0.5 * period
        candidates: list[tuple[float, float | None, float, float]] = []
        for fraction in (0.35, 0.5, 0.65, 0.8, 1.0, 1.2):
            window = duration * fraction
            in_mask = np.isfinite(flux) & (np.abs(phase) <= window / 2.0)
            out_mask = np.isfinite(flux) & (np.abs(phase) >= window)

            if int(np.sum(in_mask)) < 5 or int(np.sum(out_mask)) < 20:
                continue

            in_flux = flux[in_mask]
            out_flux = flux[out_mask]
            baseline = float(np.nanmedian(out_flux))
            in_level = float(np.nanmedian(in_flux))
            refined_depth = baseline - in_level

            if not np.isfinite(refined_depth) or refined_depth <= 0:
                continue

            out_mad = float(np.nanmedian(np.abs(out_flux - baseline)))
            in_mad = float(np.nanmedian(np.abs(in_flux - in_level)))
            out_sigma = 1.4826 * out_mad if out_mad > 0 else float(np.nanstd(out_flux))
            in_sigma = 1.4826 * in_mad if in_mad > 0 else float(np.nanstd(in_flux))
            out_sigma = out_sigma if np.isfinite(out_sigma) and out_sigma > 0 else 0.0
            in_sigma = in_sigma if np.isfinite(in_sigma) and in_sigma > 0 else 0.0

            depth_err = None
            if out_sigma > 0 or in_sigma > 0:
                depth_err = float(
                    np.sqrt(
                        (out_sigma**2 / max(int(np.sum(out_mask)), 1))
                        + (in_sigma**2 / max(int(np.sum(in_mask)), 1))
                    )
                )
            score = refined_depth / (depth_err if depth_err is not None and depth_err > 0 else refined_depth * 1e-3)
            candidates.append((refined_depth, depth_err, fraction, score))

        if not candidates:
            return fallback_depth, None, None

        candidates.sort(key=lambda item: (item[3], item[0]), reverse=True)
        refined_depth, depth_err, fraction, _score = candidates[0]
        note = f"Transit depth was refined from folded flux to {refined_depth:.6f} using a {fraction:.2f}x transit window."
        return refined_depth, depth_err, note

    @staticmethod
    def _augment_radius_metrics(
        metrics: dict[str, Any],
        stellar_radius_rsun: float | None,
        stellar_radius_err_rsun: float | None,
        planet_radius_rearth: float | None,
        planet_radius_err_rearth: float | None,
    ) -> dict[str, Any]:
        enriched = dict(metrics)
        if stellar_radius_rsun is not None:
            enriched["stellar_radius_rsun"] = round(float(stellar_radius_rsun), 5)
        if stellar_radius_err_rsun is not None:
            enriched["stellar_radius_err_rsun"] = round(float(stellar_radius_err_rsun), 5)
        if planet_radius_rearth is not None:
            enriched["planet_radius_rearth"] = round(float(planet_radius_rearth), 5)
        if planet_radius_err_rearth is not None:
            enriched["planet_radius_err_rearth"] = round(float(planet_radius_err_rearth), 5)
        return enriched

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
            reason = "The strongest BLS peak is weak above the noise floor, so this is only a tentative candidate."
            return raw_confidence_percent, False, "Weak candidate", reason, metrics

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

    def _choose_period_solution(self, periodogram: Any) -> tuple[float, float, float, str | None]:
        power = np.asarray(periodogram.power, dtype=float)
        periods = np.asarray(periodogram.period, dtype=float)
        transit_times = np.asarray(periodogram.transit_time, dtype=float)
        finite = np.isfinite(power) & np.isfinite(periods) & np.isfinite(transit_times)
        if not np.any(finite):
            raise RuntimeError("No finite periodogram peaks were available")

        finite_power = power[finite]
        max_power = float(np.nanmax(finite_power))
        argmax_index = int(np.nanargmax(power))

        near_best = finite & (power >= max_power * self.period_power_tolerance)
        if np.any(near_best):
            candidate_indices = np.where(near_best)[0]
            chosen_index = int(candidate_indices[np.argmax(periods[candidate_indices])])
            if chosen_index != argmax_index:
                note = (
                    f"Period preference selected a longer near-equal peak at {periods[chosen_index]:.6f} days "
                    f"instead of the tallest peak at {periods[argmax_index]:.6f} days."
                )
            else:
                note = None
            return (
                float(periods[chosen_index]),
                float(transit_times[chosen_index]),
                float(power[chosen_index]),
                note,
            )

        return float(periods[argmax_index]), float(transit_times[argmax_index]), float(power[argmax_index]), None

