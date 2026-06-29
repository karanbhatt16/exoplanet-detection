from __future__ import annotations

import argparse
from pathlib import Path

from exoplanet_pipeline import DatasetInspector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and preview a TESS light-curve dataset.")
    parser.add_argument("--path", type=str, help="Local file path to a CSV/FITS/Parquet dataset.")
    parser.add_argument("--target-id", type=str, help="TIC ID to download from MAST via lightkurve.")
    parser.add_argument("--sector", type=int, default=None, help="Optional TESS sector filter for remote download.")
    parser.add_argument("--output", type=str, default="artifacts/dataset_preview.png", help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Display the matplotlib plot interactively.")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the preview plot.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    inspector = DatasetInspector(target_id=args.target_id, sector=args.sector)

    if args.path:
        preview = inspector.load_local(args.path)
    elif args.target_id:
        preview = inspector.load_remote()
    else:
        parser.error("Provide either --path or --target-id")
        return 2

    print(inspector.format_summary(preview))

    if preview.dataframe is not None:
        print("\nSample rows:")
        print(inspector.preview_table(preview.dataframe))

    if not args.no_plot:
        output_path = inspector.plot_preview(preview, None if args.show else Path(args.output), show=args.show)
        if output_path is not None:
            print(f"\nSaved preview plot to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
