from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter.scrolledtext import ScrolledText

from exoplanet_pipeline import TransitSearcher


class ExoplanetApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Exoplanet Transit Search")
        self.geometry("1280x860")
        self.minsize(1100, 760)

        self.source_mode = tk.StringVar(value="target")
        self.target_id_var = tk.StringVar()
        self.file_path_var = tk.StringVar()
        self.sector_var = tk.StringVar()
        self.use_all_sectors_var = tk.BooleanVar(value=False)
        self.min_period_var = tk.StringVar(value="0.5")
        self.max_period_var = tk.StringVar(value="20.0")
        self.min_duration_var = tk.StringVar(value="0.05")
        self.max_duration_var = tk.StringVar(value="0.3")
        self.duration_steps_var = tk.StringVar(value="8")
        self.stellar_radius_var = tk.StringVar()
        self.stellar_radius_err_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        self._canvas: FigureCanvasTkAgg | None = None
        self._toolbar: NavigationToolbar2Tk | None = None
        self._current_searcher: TransitSearcher | None = None
        self._target_entry: ttk.Entry | None = None
        self._file_entry: ttk.Entry | None = None
        self._browse_button: ttk.Button | None = None
        self._run_button: ttk.Button | None = None
        self._plot_canvas: tk.Canvas | None = None
        self._plot_scrollbar: ttk.Scrollbar | None = None
        self._plot_inner_frame: ttk.Frame | None = None
        self._plot_window_id: int | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(self, text="Input")
        control_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=1)

        ttk.Radiobutton(control_frame, text="TESS Target ID", value="target", variable=self.source_mode, command=self._toggle_source_mode).grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )
        self._target_entry = ttk.Entry(control_frame, textvariable=self.target_id_var, width=32)
        self._target_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=6)

        ttk.Radiobutton(control_frame, text="Local file", value="file", variable=self.source_mode, command=self._toggle_source_mode).grid(
            row=1, column=0, sticky="w", padx=8, pady=6
        )
        self._file_entry = ttk.Entry(control_frame, textvariable=self.file_path_var)
        self._file_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        self._browse_button = ttk.Button(control_frame, text="Browse", command=self._browse_file)
        self._browse_button.grid(row=1, column=2, sticky="w", padx=8, pady=6)

        ttk.Label(control_frame, text="Sector").grid(row=0, column=2, sticky="e", padx=8, pady=6)
        ttk.Entry(control_frame, textvariable=self.sector_var, width=10).grid(row=0, column=3, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(control_frame, text="Use all sectors", variable=self.use_all_sectors_var).grid(
            row=1, column=3, sticky="w", padx=8, pady=6
        )

        settings = ttk.LabelFrame(self, text="Search Settings")
        settings.grid(row=1, column=0, sticky="ew", padx=12)
        for idx in range(8):
            settings.columnconfigure(idx, weight=1 if idx % 2 else 0)

        self._add_labeled_entry(settings, "Min period", self.min_period_var, 0, 0)
        self._add_labeled_entry(settings, "Max period", self.max_period_var, 0, 2)
        self._add_labeled_entry(settings, "Min duration", self.min_duration_var, 1, 0)
        self._add_labeled_entry(settings, "Max duration", self.max_duration_var, 1, 2)
        self._add_labeled_entry(settings, "Duration steps", self.duration_steps_var, 0, 4, width=10)
        self._add_labeled_entry(settings, "Stellar radius (Rsun)", self.stellar_radius_var, 2, 0)
        self._add_labeled_entry(settings, "Radius error (Rsun)", self.stellar_radius_err_var, 2, 2)

        self._run_button = ttk.Button(settings, text="Run Search", command=self._run_search)
        self._run_button.grid(row=0, column=6, rowspan=3, padx=12, pady=8, sticky="ns")

        output = ttk.Frame(self)
        output.grid(row=2, column=0, sticky="nsew", padx=12, pady=12)
        output.columnconfigure(0, weight=1)
        output.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(output)
        notebook.grid(row=0, column=0, sticky="nsew")

        self.results_tab = ttk.Frame(notebook)
        self.data_tab = ttk.Frame(notebook)
        self.plot_tab = ttk.Frame(notebook)
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.data_tab, text="Data")
        notebook.add(self.plot_tab, text="Plot")

        self.results_text = ScrolledText(self.results_tab, wrap="word", height=18)
        self.results_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.data_text = ScrolledText(self.data_tab, wrap="word", height=18)
        self.data_text.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_plot_area()

        status_bar = ttk.Frame(self)
        status_bar.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self._toggle_source_mode()

    def _build_plot_area(self) -> None:
        if self._plot_canvas is not None:
            self._plot_canvas.destroy()
        if self._plot_scrollbar is not None:
            self._plot_scrollbar.destroy()

        self._plot_canvas = tk.Canvas(self.plot_tab, highlightthickness=0)
        self._plot_scrollbar = ttk.Scrollbar(self.plot_tab, orient="vertical", command=self._plot_canvas.yview)
        self._plot_canvas.configure(yscrollcommand=self._plot_scrollbar.set)

        self._plot_canvas.pack(side="left", fill="both", expand=True)
        self._plot_scrollbar.pack(side="right", fill="y")

        self._plot_inner_frame = ttk.Frame(self._plot_canvas)
        self._plot_window_id = self._plot_canvas.create_window((0, 0), window=self._plot_inner_frame, anchor="nw")
        self._plot_inner_frame.bind("<Configure>", self._on_plot_frame_configure)
        self._plot_canvas.bind("<Configure>", self._on_plot_canvas_configure)
        self._plot_canvas.bind("<Enter>", self._bind_mousewheel)
        self._plot_canvas.bind("<Leave>", self._unbind_mousewheel)

        placeholder = ttk.Label(self._plot_inner_frame, text="Run a search to generate plots.")
        placeholder.pack(fill="both", expand=True, padx=8, pady=8)

    def _on_plot_frame_configure(self, _event: tk.Event) -> None:
        if self._plot_canvas is None or self._plot_inner_frame is None:
            return
        self._plot_canvas.configure(scrollregion=self._plot_canvas.bbox("all"))

    def _on_plot_canvas_configure(self, event: tk.Event) -> None:
        if self._plot_canvas is None or self._plot_window_id is None:
            return
        self._plot_canvas.itemconfigure(self._plot_window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self._plot_canvas is None:
            return
        # Windows uses 120-multiples for wheel events.
        self._plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.unbind_all("<MouseWheel>")

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, column: int, width: int = 16) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="e", padx=8, pady=6)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=column + 1, sticky="ew", padx=8, pady=6)

    def _toggle_source_mode(self) -> None:
        mode = self.source_mode.get()
        state_target = "normal" if mode == "target" else "disabled"
        state_file = "normal" if mode == "file" else "disabled"

        if self._target_entry is not None:
            self._target_entry.configure(state=state_target)
        if self._file_entry is not None:
            self._file_entry.configure(state=state_file)
        if self._browse_button is not None:
            self._browse_button.configure(state=state_file)

    def _browse_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select a light-curve file",
            filetypes=[
                ("Light-curve files", "*.csv *.txt *.parquet *.fits *.fit *.lc"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.source_mode.set("file")
            self._toggle_source_mode()

    def _run_search(self) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None

        self.results_text.delete("1.0", tk.END)
        self.data_text.delete("1.0", tk.END)
        self._rebuild_plot_area("Running search...")

        try:
            sector = int(self.sector_var.get().strip()) if self.sector_var.get().strip() else None
            min_period = float(self.min_period_var.get().strip())
            max_period = float(self.max_period_var.get().strip())
            min_duration = float(self.min_duration_var.get().strip())
            max_duration = float(self.max_duration_var.get().strip())
            duration_steps = int(self.duration_steps_var.get().strip())
            stellar_radius = float(self.stellar_radius_var.get().strip()) if self.stellar_radius_var.get().strip() else None
            stellar_radius_err = (
                float(self.stellar_radius_err_var.get().strip()) if self.stellar_radius_err_var.get().strip() else None
            )
        except ValueError as exc:
            messagebox.showerror("Invalid input", f"Please check the numeric settings.\n\n{exc}")
            return

        if self.source_mode.get() == "target":
            target_id = self.target_id_var.get().strip()
            if not target_id:
                messagebox.showerror("Missing target ID", "Please enter a TIC target ID.")
                return
            file_path = None
        else:
            file_path = self.file_path_var.get().strip()
            if not file_path:
                messagebox.showerror("Missing file", "Please choose a local light-curve file.")
                return
            target_id = None

        self._set_busy(True)
        self.status_var.set("Running search...")

        worker = threading.Thread(
            target=self._search_worker,
            kwargs=dict(
                source_mode=self.source_mode.get(),
                target_id=target_id,
                file_path=file_path,
                sector=sector,
                min_period=min_period,
                max_period=max_period,
                min_duration=min_duration,
                max_duration=max_duration,
                duration_steps=duration_steps,
                stellar_radius_rsun=stellar_radius,
                stellar_radius_err_rsun=stellar_radius_err,
                use_all_sectors=self.use_all_sectors_var.get(),
            ),
            daemon=True,
        )
        worker.start()

    def _search_worker(self, **kwargs: object) -> None:
        try:
            searcher = TransitSearcher(
                target_id=kwargs["target_id"],
                sector=kwargs["sector"],
                use_all_sectors=kwargs["use_all_sectors"],
                min_period=kwargs["min_period"],
                max_period=kwargs["max_period"],
                min_duration=kwargs["min_duration"],
                max_duration=kwargs["max_duration"],
                duration_steps=kwargs["duration_steps"],
                stellar_radius_rsun=kwargs["stellar_radius_rsun"],
                stellar_radius_err_rsun=kwargs["stellar_radius_err_rsun"],
            )

            preview_text = ""
            source_lightcurve = None

            if kwargs["source_mode"] == "file":
                preview = searcher.inspector.load_local(kwargs["file_path"])
                source_lightcurve = searcher._preview_to_lightcurve(preview)
                assessment = searcher.analyze_lightcurve(source_lightcurve, preview.source)
                preview_text = searcher.inspector.format_summary(preview)
                if preview.dataframe is not None:
                    preview_text += "\n\nSample rows:\n"
                    preview_text += searcher.inspector.preview_table(preview.dataframe)
            else:
                source_lightcurve, source = searcher.load_target()
                assessment = searcher.analyze_lightcurve(source_lightcurve, source)
                preview_summary = searcher.inspector._summarize_lightcurve(source_lightcurve, source)
                preview_text = "\n".join([f"{key}: {value}" for key, value in preview_summary.items()])

            result_text = searcher.summarize(assessment)
            self.after(
                0,
                lambda: self._display_result(
                    searcher=searcher,
                    result_text=result_text,
                    preview_text=preview_text,
                    assessment=assessment,
                    source_lightcurve=source_lightcurve,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.after(0, lambda: self._handle_error(error_text))

    def _display_result(
        self,
        searcher: TransitSearcher,
        result_text: str,
        preview_text: str,
        assessment,
        source_lightcurve,
    ) -> None:  # type: ignore[no-untyped-def]
        self._current_searcher = searcher
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, result_text)

        self.data_text.delete("1.0", tk.END)
        self.data_text.insert(tk.END, preview_text)

        try:
            figure = searcher.build_result_figure(assessment, source_lightcurve=source_lightcurve)
        except Exception as exc:  # noqa: BLE001
            self._handle_error("Could not render plot:\n\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            return

        self._clear_plot_area()
        if self._plot_inner_frame is None:
            self._build_plot_area()
        self._canvas = FigureCanvasTkAgg(figure, master=self._plot_inner_frame)
        self._canvas.draw()
        widget = self._canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        self._toolbar = NavigationToolbar2Tk(self._canvas, self._plot_inner_frame, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(fill="x", padx=8, pady=(0, 8))
        self._on_plot_frame_configure(None)

        self.status_var.set("Search complete")
        self._set_busy(False)

    def _handle_error(self, error_text: str) -> None:
        self.status_var.set("Search failed")
        self._set_busy(False)
        messagebox.showerror("Search failed", error_text)

        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, error_text)

        self._clear_plot_area()
        self._rebuild_plot_area("No plot available.")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        if self._run_button is not None:
            self._run_button.configure(state=state)

    def _clear_plot_area(self) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._plot_inner_frame is not None:
            for child in self._plot_inner_frame.winfo_children():
                child.destroy()

    def _rebuild_plot_area(self, message: str) -> None:
        self._clear_plot_area()
        if self._plot_canvas is not None:
            self._plot_canvas.destroy()
        if self._plot_scrollbar is not None:
            self._plot_scrollbar.destroy()
        self._plot_canvas = None
        self._plot_scrollbar = None
        self._plot_inner_frame = None
        self._plot_window_id = None
        self._build_plot_area()
        if self._plot_inner_frame is not None:
            for child in self._plot_inner_frame.winfo_children():
                child.destroy()
            ttk.Label(self._plot_inner_frame, text=message).pack(fill="both", expand=True, padx=8, pady=8)


def main() -> int:
    app = ExoplanetApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
