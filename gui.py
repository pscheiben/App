import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from network_engine import NetworkEngine

class GatingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("High-Speed Gating & DUT Characterization Suite v1.9")
        self.geometry("1600x950")
        self.engine = NetworkEngine()
        self.plot_options = ["Magnitude (dB)", "Impulse Location", "Impedance (Ohm)", "Phase"]
        self._setup_ui()

    def _setup_ui(self):
        controls = ttk.Frame(self, padding="10")
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(controls, text="Load Touchstone File", command=self._load).pack(fill=tk.X, pady=5)
        
        ttk.Label(controls, text="Gate Center (ns):").pack(anchor="w")
        self.ent_center = ttk.Entry(controls); self.ent_center.insert(0, "2.5"); self.ent_center.pack(pady=2)

        ttk.Label(controls, text="Gate Span (ns):").pack(anchor="w")
        self.ent_span = ttk.Entry(controls); self.ent_span.insert(0, "5.0"); self.ent_span.pack(pady=2)

        ttk.Label(controls, text="Reference Z0 (Ω):").pack(anchor="w", pady=(10, 0))
        self.ent_z0 = ttk.Entry(controls); self.ent_z0.insert(0, "50.0"); self.ent_z0.pack(pady=2)

        ttk.Label(controls, text="Permittivity (εr):").pack(anchor="w")
        self.ent_er = ttk.Entry(controls); self.ent_er.insert(0, "4.4"); self.ent_er.pack(pady=2)

        self.axis_unit = tk.StringVar(value="Time")
        ttk.Radiobutton(controls, text="X-Axis: Time (ns)", variable=self.axis_unit, value="Time").pack(anchor="w")
        ttk.Radiobutton(controls, text="X-Axis: Distance (mm)", variable=self.axis_unit, value="Dist").pack(anchor="w")

        self.show_markers = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto-Detect Features", variable=self.show_markers).pack(anchor="w", pady=5)

        self.quad_configs = []
        for i in range(4):
            frame = ttk.LabelFrame(controls, text=f"Quadrant {i+1}", padding="5")
            frame.pack(fill=tk.X, pady=5)
            p_sel = ttk.Combobox(frame); p_sel.pack(fill=tk.X)
            v_sel = ttk.Combobox(frame, values=self.plot_options); v_sel.current(i); v_sel.pack(fill=tk.X)
            self.quad_configs.append({'param': p_sel, 'view': v_sel})

        ttk.Button(controls, text="Update All Diagrams", command=self._update).pack(fill=tk.X, pady=20)

        plot_container = ttk.Frame(self)
        plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        NavigationToolbar2Tk(self.canvas, plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _load(self):
        path = filedialog.askopenfilename(filetypes=[("Touchstone", "*.s2p *.s4p")])
        if path:
            self.engine.load_file(path)
            all_p = self.engine.get_available_params()
            for config in self.quad_configs:
                config['param']['values'] = all_p
                config['param'].current(0)
            self._update()

    def _update(self):
        if not self.engine.network: return
        c, s = float(self.ent_center.get()), float(self.ent_span.get())
        z0, er = float(self.ent_z0.get()), float(self.ent_er.get())
        use_dist = (self.axis_unit.get() == "Dist")
        self.fig.clear()
        
        for i, config in enumerate(self.quad_configs):
            ax = self.fig.add_subplot(2, 2, i+1)
            p, v = config['param'].get(), config['view'].get()
            orig, gated, t_ns, z_ohm, dist_mm = self.engine.get_processed_data(p, c, s, z0, er)
            x_data = dist_mm if use_dist else t_ns
            x_unit = "mm" if use_dist else "ns"
            ax.grid(True, alpha=0.3)

            if v == "Magnitude (dB)":
                orig.plot_s_db(ax=ax, label='Raw'); gated.plot_s_db(ax=ax, label='Gated')
                ax.set_title(f"{p} Mag (Ref: {z0}Ω)")
            elif v == "Impulse Location":
                # Detect the main energy arrival peak
                mag_td = np.abs(orig.s_time[:,0,0])
                idx_peak = np.argmax(mag_td)
                orig.plot_s_db_time(ax=ax)
                if self.show_markers.get():
                    ax.axvline(x_data[idx_peak], color='green', linestyle=':', label='Arrival')
                    ax.text(x_data[idx_peak], -30, f"Arrival\n{x_data[idx_peak]:.2f}{x_unit}", 
                            color='green', fontsize=8, ha='left')
                ax.axvspan(c-s/2, c+s/2, color='orange', alpha=0.1)
                ax.set_title(f"{p} Impulse (Time of Flight)")
            elif v == "Impedance (Ohm)":
                if any(x in p for x in ["11", "22", "33", "44"]):
                    ax.plot(x_data, z_ohm)
                    if self.show_markers.get():
                        grad = np.gradient(z_ohm)
                        idx_fall = np.argmin(grad)
                        idx_rise = np.argmax(grad)
                        for idx, lbl in zip([idx_fall, idx_rise], ["Entry", "Rise"]):
                            ax.axvline(x_data[idx], color='red', linestyle='--', alpha=0.6)
                    ax.set_ylabel("Ω"); ax.set_xlabel(f"Axis: {x_unit}"); ax.set_ylim(z0-10, z0+10)
                ax.set_title(f"{p} Impedance Profile")
            elif v == "Phase":
                ax.plot(orig.frequency.f, np.unwrap(np.angle(orig.s[:,0,0], deg=True)))
                ax.set_title(f"{p} Unwrapped Phase")

        self.fig.tight_layout()
        self.canvas.draw()