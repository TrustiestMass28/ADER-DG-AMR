import yt
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from yt.units import dimensions

warnings.filterwarnings("ignore")
yt.utilities.logger.colorize_logging()


# ======================================================================
# Module-level constants and worker functions (picklable for multiprocessing)
# ======================================================================

UNITS_OVERRIDE = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
}

FIELD_INFO = {
    "Advection": {
        0: ("density_x_{m}", r"$\rho(\mathbf{x})\ $ ($\dfrac{kg}{m^3}$)",
            r"Density $\rho(\mathbf{x})$"),
    },
    "Compressible_Euler_2D": {
        0: ("mass_density_{m}", r"$\rho(\mathbf{x})\ $ ($\dfrac{kg}{m^3}$)",
            r"Density $\rho(\mathbf{x})$"),
        1: ("momentum_x_{m}", r"$u_1(\mathbf{x})\ $ ($\dfrac{m}{s}$)",
            r"Velocity $u_1(\mathbf{x})$"),
        2: ("momentum_y_{m}", r"$u_2(\mathbf{x})\ $ ($\dfrac{m}{s}$)",
            r"Velocity $u_2(\mathbf{x})$"),
        3: ("energy_density_{m}", r"$e(\mathbf{x})\ $ ($\dfrac{J}{kg}$)",
            r"Specific Energy $e(\mathbf{x})$"),
        4: ("angular_momentum_z_{m}", r"$L_z(\mathbf{x})\ $ ($\dfrac{m^2}{s}$)",
            r"Angular momentum $L_z(\mathbf{x})$"),
    },
}


def _worker_load(cfg, tstep, q):
    path = os.path.join(cfg["data_dir"], f"{cfg['prefix']}_{tstep}_q_{q}_plt")
    return yt.load(path, unit_system="mks", units_override=UNITS_OVERRIDE)


def _worker_prepare_field(ds, cfg, tstep):
    """Return (field_to_plot, field_name, ds) — standalone for workers."""
    sol_n = cfg["sol_n"]
    mode_n = cfg["mode_n"]
    field_composite = cfg["field_composite"]
    equation_type = cfg["equation_type"]

    field_to_plot = (ds.field_list[mode_n][0], ds.field_list[mode_n][-1])
    field_name = ds.field_list[mode_n][-1]

    if equation_type == "Compressible_Euler_2D" and sol_n != 0:
        density_ds = _worker_load(cfg, tstep, q=0)
        density_field_name = (
            density_ds.field_list[mode_n][0],
            density_ds.field_list[mode_n][-1],
        )

        def _field_mode_density(field, data):
            return (density_ds.r[density_field_name]
                    * density_ds.unit_system["density"])

        ds.add_field(
            name=density_field_name, function=_field_mode_density,
            sampling_type="cell", dimensions=dimensions.density,
            units=ds.unit_system["density"],
        )

        if sol_n in (1, 2):
            momentum_field_name = (
                ds.field_list[mode_n][0], ds.field_list[mode_n][-1],
            )

            def _field_mode_velocity(field, data):
                return (
                    ds.r[momentum_field_name]
                    * (ds.unit_system["momentum"] / ds.unit_system["volume"])
                    / ds.r[density_field_name]
                )

            vel_composite = field_composite.replace("momentum", "velocity")
            vel_derived = (ds.field_list[mode_n][0], vel_composite)
            ds.add_field(
                name=vel_derived, function=_field_mode_velocity,
                sampling_type="cell", dimensions=dimensions.velocity,
                units=ds.unit_system["velocity"],
            )
            field_to_plot = vel_derived
            field_name = vel_composite

        elif sol_n == 3:
            energy_density_field_name = (
                ds.field_list[mode_n][0], ds.field_list[mode_n][-1],
            )

            def _field_mode_energy(field, data):
                return (
                    ds.r[energy_density_field_name]
                    * (ds.unit_system["specific_energy"]
                       * ds.unit_system["density"])
                    / ds.r[density_field_name]
                )

            eng_composite = field_composite.replace("density_", "")
            eng_derived = (ds.field_list[mode_n][0], eng_composite)
            ds.add_field(
                name=eng_derived, function=_field_mode_energy,
                sampling_type="cell", dimensions=dimensions.specific_energy,
                units=ds.unit_system["specific_energy"],
            )
            field_to_plot = eng_derived
            field_name = eng_composite

    return field_to_plot, field_name, ds


def _bounds_worker(args):
    """Compute (min, max) for a single timestep. Returns (lo, hi) or None."""
    ts, cfg = args
    try:
        ds = _worker_load(cfg, ts, cfg["sol_n"])
        _, field_name, ds = _worker_prepare_field(ds, cfg, ts)
        lo, hi = ds.all_data().quantities.extrema(field_name)
        return float(lo), float(hi)
    except Exception as e:
        print(f"  [skip tstep {ts}] {e}")
        return None


def _plot_worker(args):
    """Load, render, and save a single timestep plot."""
    ts, idx, total, cfg = args
    path = os.path.join(
        cfg["data_dir"], f"{cfg['prefix']}_{ts}_q_{cfg['sol_n']}_plt",
    )
    if not os.path.exists(path):
        print(f"  [{idx+1}/{total}] tstep {ts}: file not found, skipping")
        return

    print(f"  [{idx+1}/{total}] tstep {ts} ...")

    ds = _worker_load(cfg, ts, cfg["sol_n"])
    field_to_plot, field_name, ds = _worker_prepare_field(ds, cfg, ts)

    sl = yt.SlicePlot(ds, "z", field_to_plot, origin="native")

    if cfg["min_val"] is not None:
        sl.set_zlim(field=field_to_plot,
                    zmin=cfg["min_val"], zmax=cfg["max_val"])
    sl.set_log(field_name, log=False)
    sl.set_cmap(field=field_name, cmap=cfg["cmap"])

    if not cfg.get("show_plot_info", True):
        sl.hide_colorbar()
        sl.hide_axes(draw_frame=False)
    else:
        sl.set_colorbar_label(field_name, cfg["label_cb"])
        sl.set_colorbar_minorticks("all", True)
        sl.show_colorbar()
        sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
        sl.annotate_title(cfg["label_title"])
        sl.set_font_size(cfg["font_size"])
        sl.set_axes_unit("m")

    if cfg["show_grids"]:
        sl.annotate_grids(alpha=1.0, linewidth=1.0)
    if cfg["show_cell_edges"]:
        sl.annotate_cell_edges(line_width=0.001, alpha=0.6, color="grey")

    sl.set_buff_size(cfg["dpi"] * cfg["fig_size"])
    sl.set_figure_size(cfg["fig_size"])
    sl.render()

    out_path = os.path.join(
        cfg["plot_dir"], f"{ts}_sol_{cfg['sol_n']}.png",
    )
    sl.save(out_path)


# ======================================================================
# Main class
# ======================================================================

class SimPlotter:
    """Plotter for ADER-DG-AMR simulation output.

    Parameters
    ----------
    data_dir : str
        Path to the "Simulation Data" directory containing plotfiles.
    plot_dir : str
        Path to the output directory for generated plots.
    equation_type : str
        One of "Compressible_Euler_2D" or "Advection".
    prefix : str
        Plotfile naming prefix (default "tstep").
    n_workers : int
        Number of parallel workers (default 4). Set to 1 for serial.
    """

    def __init__(self, data_dir="../../Results/Simulation Data",
                 plot_dir="../../Results/Plots",
                 equation_type="Compressible_Euler_2D",
                 prefix="tstep",
                 n_workers=4):
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        self.equation_type = equation_type
        self.prefix = prefix
        self.n_workers = n_workers
        os.makedirs(self.plot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Timestep discovery
    # ------------------------------------------------------------------
    def discover_timesteps(self, q=0):
        """Scan data_dir and return sorted list of all available timestep
        indices for solution component q."""
        pattern = re.compile(
            rf"^{re.escape(self.prefix)}_(\d+)_q_{q}_plt$"
        )
        steps = []
        for name in os.listdir(self.data_dir):
            m = pattern.match(name)
            if m:
                steps.append(int(m.group(1)))
        steps.sort()
        return steps

    def select_timesteps(self, mode="all", steps=None, n_max=None,
                         interval=1, q=0):
        """Build a list of timestep indices to plot.

        Parameters
        ----------
        mode : str
            "all"      - read from data dir, plot every available step.
            "list"     - use the explicit *steps* list.
            "sequence" - generate range(0, n_max+1, interval).
        steps : list[int], optional
            Explicit timestep list (mode="list").
        n_max : int, optional
            Upper bound for generated sequence (mode="sequence").
        interval : int, optional
            Spacing for generated sequence (mode="sequence", default 1).
        q : int
            Solution component used for discovery (mode="all").
        """
        if mode == "all":
            return self.discover_timesteps(q=q)
        elif mode == "list":
            if steps is None:
                raise ValueError("mode='list' requires steps=[...]")
            return sorted(steps)
        elif mode == "sequence":
            if n_max is None:
                raise ValueError("mode='sequence' requires n_max")
            return list(range(0, n_max + 1, interval))
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    # ------------------------------------------------------------------
    # Config dict for workers
    # ------------------------------------------------------------------
    def _make_cfg(self, sol_n, mode_n, **extra):
        tpl, label_cb, label_title = FIELD_INFO[self.equation_type][sol_n]
        cfg = {
            "data_dir": self.data_dir,
            "plot_dir": self.plot_dir,
            "prefix": self.prefix,
            "equation_type": self.equation_type,
            "sol_n": sol_n,
            "mode_n": mode_n,
            "field_composite": tpl.format(m=mode_n),
            "label_cb": label_cb,
            "label_title": label_title,
        }
        cfg.update(extra)
        return cfg

    # ------------------------------------------------------------------
    # Colorbar bounds (parallel)
    # ------------------------------------------------------------------
    def compute_colorbar_bounds(self, sol_n, mode_n, timesteps,
                                sample_every=1):
        """Compute global (min, max) across timesteps using parallel workers."""
        cfg = self._make_cfg(sol_n, mode_n)
        sampled = timesteps[::sample_every]
        work = [(ts, cfg) for ts in sampled]

        mins, maxs = [], []
        if self.n_workers > 1 and len(work) > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                for result in pool.map(_bounds_worker, work):
                    if result is not None:
                        mins.append(result[0])
                        maxs.append(result[1])
        else:
            for item in work:
                result = _bounds_worker(item)
                if result is not None:
                    mins.append(result[0])
                    maxs.append(result[1])

        if not mins:
            raise RuntimeError("No valid data found for colorbar bounds")
        return min(mins), max(maxs)

    # ------------------------------------------------------------------
    # Plotting (parallel)
    # ------------------------------------------------------------------
    def plot(self, sol_n=0, mode_n=0, mode="all", steps=None,
             n_max=None, interval=1, cmap="inferno", dpi=300,
             fig_size=5, font_size=20, show_grids=False,
             show_cell_edges=True, fixed_bounds=True,
             cb_sample_every=1, show_plot_info=True):
        """Main entry point: select timesteps and produce plots.

        Parameters
        ----------
        sol_n : int
            Solution component index (0=density, 1=mom_x, ...).
        mode_n : int
            Mode index to plot (usually 0 for cell average).
        mode : str
            Timestep selection mode: "all", "list", or "sequence".
        steps : list[int]
            Explicit timestep list (mode="list").
        n_max : int
            Max timestep (mode="sequence").
        interval : int
            Timestep spacing (mode="sequence").
        cmap : str
            Matplotlib/yt colormap name.
        dpi : int
            Output resolution.
        fig_size : int
            Figure size in inches.
        font_size : int
            Annotation font size.
        show_grids : bool
            Annotate AMR grid patches.
        show_cell_edges : bool
            Draw cell edges.
        fixed_bounds : bool
            Use global min/max for colorbar across all timesteps.
        cb_sample_every : int
            Sample every N-th step when computing colorbar bounds.
        """
        timesteps = self.select_timesteps(
            mode=mode, steps=steps, n_max=n_max, interval=interval, q=sol_n,
        )
        if not timesteps:
            print("No timesteps to plot.")
            return

        print(f"Plotting sol_n={sol_n}, mode_n={mode_n}, "
              f"{len(timesteps)} timestep(s): "
              f"[{timesteps[0]} ... {timesteps[-1]}]")

        # Colorbar bounds (parallel)
        min_val, max_val = None, None
        if fixed_bounds:
            print("Computing colorbar bounds ...")
            min_val, max_val = self.compute_colorbar_bounds(
                sol_n, mode_n, timesteps, sample_every=cb_sample_every,
            )
            print(f"  range: [{min_val}, {max_val}]")

        # Build config dict for plot workers
        cfg = self._make_cfg(
            sol_n, mode_n,
            cmap=cmap, dpi=dpi, fig_size=fig_size, font_size=font_size,
            show_grids=show_grids, show_cell_edges=show_cell_edges,
            min_val=min_val, max_val=max_val, show_plot_info=show_plot_info,
        )

        total = len(timesteps)
        work = [(ts, idx, total, cfg) for idx, ts in enumerate(timesteps)]

        # Plot (parallel)
        if self.n_workers > 1 and len(work) > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = [pool.submit(_plot_worker, w) for w in work]
                for f in as_completed(futures):
                    f.result()  # re-raise any exception
        else:
            for item in work:
                _plot_worker(item)

        print("Done.")


# ======================================================================
# CLI
# ======================================================================

def main():
    plotter = SimPlotter(
        data_dir="../../Results/Simulation Data",
        plot_dir="../../Results/Plots",
        equation_type="Compressible_Euler_2D",
        n_workers=4,      # set to 1 for serial execution
    )

    # Toggle overlays
    show_amr_patches = False   # AMR patch boundaries
    show_cell_edges  = False   # individual cell contours
    show_plot_info   = False    # False = data only (no colorbar, axes, title, timestamp)

    # Plot ALL available timesteps for density (sol_n=0)
    # plotter.plot(sol_n=0, mode="all", show_grids=show_amr_patches, show_cell_edges=show_cell_edges, show_plot_info=show_plot_info)

    # Plot specific timesteps
    plotter.plot(sol_n=0, mode="all", steps=[3156], show_grids=show_amr_patches, show_cell_edges=show_cell_edges, show_plot_info=show_plot_info)

    # Plot a generated sequence: 0, 100, 200, ..., 2000
    # plotter.plot(sol_n=0, mode="sequence", n_max=2000, interval=100, show_grids=show_amr_patches, show_cell_edges=show_cell_edges, show_plot_info=show_plot_info)


if __name__ == "__main__":
    main()
