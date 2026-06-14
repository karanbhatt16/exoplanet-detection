from __future__ import annotations

import argparse
from pathlib import Path

from exoplanet_pipeline.transit_search import TransitSearcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search for transit candidates using BLS on long-baseline light curves.")
    parser.add_argument("--path", type=str, help="Local file path to a CSV/FITS/Parquet light curve.")
    parser.add_argument("--target-id", type=str, help="TIC ID to download from MAST via lightkurve.")
    parser.add_argument("--sector", type=int, default=None, help="Optional TESS sector filter.")
    parser.add_argument("--min-period", type=float, default=0.5, help="Minimum trial period in days.")
    parser.add_argument("--all-sectors", action="store_true", help="Ignore sector filtering and stitch all available TESS sectors for the longest baseline.")
    parser.add_argument("--max-period", type=float, default=20.0, help="Maximum trial period in days.")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Minimum trial duration in days.")
    parser.add_argument("--max-duration", type=float, default=0.3, help="Maximum trial duration in days.")
    parser.add_argument("--duration-steps", type=int, default=8, help="Number of trial durations for BLS search.")
    parser.add_argument("--output", type=str, default="artifacts/transit_candidate.png", help="Output plot path.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    searcher = TransitSearcher(
        target_id=args.target_id,
        sector=args.sector,
        use_all_sectors=args.all_sectors,
        min_period=args.min_period,
        max_period=args.max_period,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        duration_steps=args.duration_steps,
    )

    try:
        if args.path:
            candidate = searcher.analyze_path(args.path)
        elif args.target_id:
            candidate = searcher.analyze_remote()
        else:
            parser.error("Provide either --path or --target-id")
            return 2
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(searcher.summarize(candidate))

    if not args.no_plot:
        output_path = searcher.plot_candidate(candidate, None if args.show else Path(args.output), show=args.show)
        if output_path is not None:
            print(f"\nSaved transit search plot to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
