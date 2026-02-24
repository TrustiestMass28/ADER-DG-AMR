import yt
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from yt.units import dimensions

warnings.filterwarnings("ignore")
yt.set_log_level("error")


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

    opts = dict(
        sol_n=0, mode_n=0, mode="all", steps=[1580],
        show_grids=False, show_cell_edges=False, show_plot_info=False,
    )

    # Full domain view
    plotter.plot(**opts, view="domain")

    # Zoomed detail view
    plotter.plot(**opts, view="detail",
                 zoom_center=(0.75, 0.75), zoom_factor=4.0)

    # Build GIFs into Doc/media
    media_dir = "../../Doc/media"
    plotter.make_gif(sol_n=0, view="domain", fps=10,
                     output_name="kh_domain.gif", output_dir=media_dir)
    plotter.make_gif(sol_n=0, view="detail", fps=10,
                     output_name="kh_detail.gif", output_dir=media_dir)


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
        #       (field_template,       label_cb,                                        label_title)
        0: ("density_x_{m}",     r"$\rho(\mathbf{x})\ $ ($\dfrac{kg}{m^3}$)",
            r"Density $\rho(\mathbf{x})$"),
    },
    "Compressible_Euler_2D": {
        0: ("mass_density_{m}",     r"$\rho(\mathbf{x})\ $ ($\dfrac{kg}{m^3}$)",
            r"Density $\rho(\mathbf{x})$"),
        1: ("momentum_x_{m}",      r"$u_1(\mathbf{x})\ $ ($\dfrac{m}{s}$)",
            r"Velocity $u_1(\mathbf{x})$"),
        2: ("momentum_y_{m}",      r"$u_2(\mathbf{x})\ $ ($\dfrac{m}{s}$)",
            r"Velocity $u_2(\mathbf{x})$"),
        3: ("energy_density_{m}",  r"$e(\mathbf{x})\ $ ($\dfrac{J}{kg}$)",
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


def _make_slice(ds, field_to_plot, field_name, cfg, with_overlays,
                show_timestamp=True, zoom_center=None, zoom_factor=None):
    """Create a configured yt SlicePlot.

    Parameters
    ----------
    with_overlays : bool
        If True, add colorbar, axes, title, font, grids, cell edges.
        If False, produce a clean data-only image (no axes, no colorbar).
    show_timestamp : bool
        Show timestamp annotation (only when with_overlays is True).
    zoom_center : tuple(float, float) or None
        (x, y) center for the zoomed view.
    zoom_factor : float or None
        Zoom factor (e.g. 2.0 = show half the domain width in each direction).
    """
    sl = yt.SlicePlot(ds, "z", field_to_plot, origin="native")

    if zoom_center is not None and zoom_factor is not None:
        le = ds.domain_left_edge.d
        re = ds.domain_right_edge.d
        wx = (re[0] - le[0]) / zoom_factor
        wy = (re[1] - le[1]) / zoom_factor
        sl.set_center(list(zoom_center), unit="m")
        sl.set_width(((wx, "m"), (wy, "m")))

    if cfg["min_val"] is not None:
        sl.set_zlim(field=field_to_plot,
                    zmin=cfg["min_val"], zmax=cfg["max_val"])
    sl.set_log(field_name, log=False)
    sl.set_cmap(field=field_name, cmap=cfg["cmap"])

    if with_overlays:
        sl.set_colorbar_label(field_name, cfg["label_cb"])
        sl.set_colorbar_minorticks("all", True)
        sl.show_colorbar()
        if show_timestamp:
            sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
        sl.annotate_title(cfg["label_title"])
        sl.set_font_size(cfg["font_size"])
        sl.set_axes_unit("m")
        if cfg["show_grids"]:
            sl.annotate_grids(alpha=1.0, linewidth=1.0)
        if cfg["show_cell_edges"]:
            sl.annotate_cell_edges(line_width=0.001, alpha=0.6, color="grey")
    else:
        sl.hide_colorbar()
        sl.hide_axes(draw_frame=False)

    sl.set_buff_size(int(cfg["dpi"] * cfg["fig_size"]))
    sl.set_figure_size(cfg["fig_size"])
    sl.render()

    # Shrink tick labels to avoid overlap at corners
    if with_overlays:
        tick_size = cfg["font_size"] * 0.6
        for p in sl.plots.values():
            p.axes.tick_params(labelsize=tick_size)

    return sl


def _render_row(cfg, ts, field_to_plot, field_name, show_timestamp=True,
                zoom_center=None, zoom_factor=None):
    """Render a pair of panels (left=cell edges, right=clean) and return
    their image arrays.  When zoom_center/zoom_factor are given the view
    is zoomed in with proportionally higher resolution."""
    import matplotlib.image as mpimg

    zoom_kw = dict(zoom_center=zoom_center, zoom_factor=zoom_factor)
    tag = "full" if zoom_factor is None else "zoom"

    # Left: cell edges + overlays
    ds_l = _worker_load(cfg, ts, cfg["sol_n"])
    _, _, ds_l = _worker_prepare_field(ds_l, cfg, ts)
    cfg_l = dict(cfg, show_cell_edges=True, show_grids=False)
    sl_l = _make_slice(ds_l, field_to_plot, field_name,
                       cfg_l, with_overlays=True,
                       show_timestamp=show_timestamp, **zoom_kw)
    tmp_l = os.path.join(cfg["plot_dir"], f"_tmp_{tag}_left_{ts}.png")
    sl_l.save(tmp_l)
    tmp_l = _find_saved(tmp_l)

    # Right: clean (no cell edges, no timestamp)
    ds_r = _worker_load(cfg, ts, cfg["sol_n"])
    _, _, ds_r = _worker_prepare_field(ds_r, cfg, ts)
    cfg_r = dict(cfg, show_cell_edges=False, show_grids=False)
    sl_r = _make_slice(ds_r, field_to_plot, field_name,
                       cfg_r, with_overlays=True,
                       show_timestamp=False, **zoom_kw)
    tmp_r = os.path.join(cfg["plot_dir"], f"_tmp_{tag}_right_{ts}.png")
    sl_r.save(tmp_r)
    tmp_r = _find_saved(tmp_r)

    img_l = mpimg.imread(tmp_l)
    img_r = mpimg.imread(tmp_r)
    os.remove(tmp_l)
    os.remove(tmp_r)
    return img_l, img_r


def _plot_worker(args):
    """Load, render, and save a single timestep plot."""
    import matplotlib.pyplot as plt

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

    view = cfg.get("view", "domain")

    if view in ("domain", "detail"):
        zoom_kw = {}
        if view == "detail":
            zoom_kw = dict(zoom_center=cfg["zoom_center"],
                           zoom_factor=cfg["zoom_factor"])

        img_l, img_r = _render_row(cfg, ts, field_to_plot, field_name,
                                   show_timestamp=True,
                                   **zoom_kw)
        _, w_l = img_l.shape[:2]
        _, w_r = img_r.shape[:2]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(cfg["fig_size"] * 2.4, cfg["fig_size"]),
            gridspec_kw={"width_ratios": [w_l, w_r]},
        )
        ax1.imshow(img_l); ax1.axis("off")
        ax2.imshow(img_r); ax2.axis("off")
        fig.tight_layout(pad=0.2)

        out_path = os.path.join(
            cfg["plot_dir"], f"{ts}_sol_{cfg['sol_n']}_{view}.png",
        )
        fig.savefig(out_path, dpi=cfg["dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        show_overlays = cfg.get("show_plot_info", True)
        sl = _make_slice(ds, field_to_plot, field_name, cfg,
                         with_overlays=show_overlays)
        out_path = os.path.join(
            cfg["plot_dir"], f"{ts}_sol_{cfg['sol_n']}.png",
        )
        sl.save(out_path)


def _find_saved(base_path):
    """yt.save() may append extra info to the filename. Find the actual file."""
    if os.path.isfile(base_path):
        return base_path
    directory = os.path.dirname(base_path)
    basename = os.path.basename(base_path).replace(".png", "")
    for f in os.listdir(directory):
        if f.startswith(basename) and f.endswith(".png"):
            return os.path.join(directory, f)
    return base_path


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
        elif mode == "last":
            all_steps = self.discover_timesteps(q=q)
            if not all_steps:
                return []
            return [all_steps[-1]]
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
             cb_sample_every=1, show_plot_info=True,
             view="domain",
             zoom_center=None, zoom_factor=None):
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

        buff = dpi * fig_size
        print(f"Plotting sol_n={sol_n}, mode_n={mode_n}, "
              f"{len(timesteps)} timestep(s): "
              f"[{timesteps[0]} ... {timesteps[-1]}]")
        print(f"  fig_size={fig_size}in, dpi={dpi}, "
              f"buff_size={buff}px ({buff}x{buff})")

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
            view=view,
            zoom_center=zoom_center, zoom_factor=zoom_factor,
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

    # ------------------------------------------------------------------
    # GIF generation
    # ------------------------------------------------------------------
    def make_gif(self, sol_n=0, view="domain", fps=10,
                 output_name=None, output_dir=None):
        """Build an animated GIF from previously saved plot PNGs.

        Parameters
        ----------
        sol_n : int
            Solution component (must match the plots already generated).
        view : str
            "domain" or "detail" — selects which set of PNGs to use.
        fps : int
            Frames per second for the GIF.
        output_name : str, optional
            Output filename. Defaults to ``sol_{sol_n}_{view}.gif``.
        output_dir : str, optional
            Directory for the GIF. Defaults to plot_dir.
        """
        from PIL import Image as PILImage

        suffix = f"_{view}"
        pattern = re.compile(
            rf"^(\d+)_sol_{sol_n}{suffix}\.png$"
        )

        # Collect and sort frames by timestep number
        frames_map = {}
        for fname in os.listdir(self.plot_dir):
            m = pattern.match(fname)
            if m:
                frames_map[int(m.group(1))] = fname
        if not frames_map:
            print(f"No PNG frames found for sol_n={sol_n}, view={view}")
            return

        sorted_steps = sorted(frames_map.keys())
        print(f"Building GIF: {len(sorted_steps)} frames, {fps} fps ...")

        frames = []
        for ts in sorted_steps:
            path = os.path.join(self.plot_dir, frames_map[ts])
            img = PILImage.open(path).convert("RGB")
            frames.append(img)

        if output_name is None:
            output_name = f"sol_{sol_n}_{view}.gif"
        gif_dir = output_dir if output_dir is not None else self.plot_dir
        os.makedirs(gif_dir, exist_ok=True)
        out_path = os.path.join(gif_dir, output_name)

        duration_ms = int(1000 / fps)
        frames[0].save(
            out_path, save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
