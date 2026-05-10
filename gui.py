import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import traceback
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from network_engine import NetworkEngine

class GatingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("High-Speed Gating Suite v2.3")
        self.geometry("1600x950")
        self.engine = NetworkEngine()
        self.plot_options = ["Magnitude (dB)", "Impulse Location", "Impedance (Ohm)", "Phase"]
        self._setup_ui()

    def _setup_ui(self):
        controls = ttk.Frame(self, padding="10")
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(controls, text="Load Touchstone File", command=self._load).pack(fill=tk.X, pady=5)
        
        ttk.Label(controls, text="Gate Center (ns):").pack(anchor="w")
        self.ent_center = ttk.Entry(controls)
        self.ent_center.insert(0, "2.5")
        self.ent_center.pack(pady=2)

        ttk.Label(controls, text="Gate Span (ns):").pack(anchor="w")
        self.ent_span = ttk.Entry(controls)
        self.ent_span.insert(0, "4.0")
        self.ent_span.pack(pady=2)

        ttk.Label(controls, text="Reference Z0 (Ω):").pack(anchor="w", pady=(10, 0))
        self.ent_z0 = ttk.Entry(controls)
        self.ent_z0.insert(0, "50.0")
        self.ent_z0.pack(pady=2)

        self.show_markers = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto-Detect Features", variable=self.show_markers).pack(anchor="w", pady=10)

        self.quad_configs = []
        for i in range(4):
            frame = ttk.LabelFrame(controls, text=f"Quadrant {i+1}", padding="5")
            frame.pack(fill=tk.X, pady=5)
            p_sel = ttk.Combobox(frame)
            p_sel.pack(fill=tk.X)
            v_sel = ttk.Combobox(frame, values=self.plot_options)
            v_sel.current(i)
            v_sel.pack(fill=tk.X)
            self.quad_configs.append({'param': p_sel, 'view': v_sel})

        ttk.Button(controls, text="Update Diagrams", command=self._update).pack(fill=tk.X, pady=20)

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
        
        try:
            c = float(self.ent_center.get())
            s = float(self.ent_span.get())
            z0 = float(self.ent_z0.get())
        except ValueError:
            print("Input Error: Please ensure Gate and Z0 values are numbers.")
            return

        self.fig.clear()
        
        for i, config in enumerate(self.quad_configs):
            ax = self.fig.add_subplot(2, 2, i+1)
            p = config['param'].get()
            v = config['view'].get()
            
            if not p: continue

            try:
                orig, gated, t_ns, z_ohm, y_imp = self.engine.get_processed_data(p, c, s, z0)
                ax.grid(True, alpha=0.3)

                if v == "Magnitude (dB)":
                    orig.plot_s_db(ax=ax, label='Raw')
                    gated.plot_s_db(ax=ax, label='Gated')
                    ax.set_title(f"{p} Mag (Ref: {z0}Ω)")
                    
                elif v == "Impulse Location":
                    orig.plot_s_db_time(ax=ax)
                    if self.show_markers.get():
                        pos_mask = t_ns > 0
                        if np.any(pos_mask):
                            valid_t = t_ns[pos_mask]
                            valid_mag = np.abs(y_imp)[pos_mask]
                            idx_peak = np.argmax(valid_mag)
                            peak_time = valid_t[idx_peak]
                            
                            ax.axvline(peak_time, color='green', linestyle='--', alpha=0.8)
                            ax.text(peak_time + 0.3, -30, f"Arrival: {peak_time:.2f} ns", 
                                    color='green', fontsize=9, fontweight='bold', ha='left',
                                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
                    
                    ax.axvspan(c-s/2, c+s/2, color='orange', alpha=0.1)
                    ax.set_title(f"{p} Impulse (Time of Flight)")
                    
                elif v == "Impedance (Ohm)":
                    if any(x in p for x in ["11", "22", "33", "44"]):
                        ax.plot(t_ns, z_ohm)
                        if self.show_markers.get():
                            grad = np.gradient(z_ohm)
                            
                            # 1. Find the sharpest drop (Entry)
                            idx_fall = np.argmin(grad)
                            t_start = t_ns[idx_fall]
                            
                            # 2. Create a "Blind Spot" mask to ignore the immediate ringing
                            # We force the algorithm to only look for the rising edge at least 0.5ns later
                            search_mask = t_ns > (t_start + 0.5)
                            
                            if np.any(search_mask):
                                valid_indices = np.where(search_mask)[0]
                                # 3. Find the sharpest rise ONLY within the masked safe zone
                                idx_rise = valid_indices[np.argmax(grad[search_mask])]
                            else:
                                idx_rise = np.argmax(grad)
                                
                            t_end = t_ns[idx_rise]
                            
                            # Draw boundary lines
                            ax.axvline(t_start, color='red', linestyle='--', alpha=0.6)
                            ax.axvline(t_end, color='red', linestyle='--', alpha=0.6)
                            
                            # Highlight the dip width
                            if t_end > t_start:
                                ax.axvspan(t_start, t_end, color='red', alpha=0.1)
                                dip_width = t_end - t_start
                                ax.text(t_start + dip_width/2, z0 - 8, f"Width: {dip_width:.2f} ns", 
                                        color='red', fontsize=9, fontweight='bold', ha='center',
                                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
                                
                        ax.set_ylabel("Ω")
                        ax.set_xlabel("Time (ns)")
                        ax.set_ylim(z0-10, z0+10)
                    else:
                        ax.text(0.5, 0.5, "Use Reflection for Z-Profile", ha='center', va='center')
                    ax.set_title(f"{p} Impedance Profile")
                    
                elif v == "Phase":
                    ax.plot(orig.frequency.f, np.unwrap(np.angle(orig.s[:,0,0], deg=True)))
                    ax.set_title(f"{p} Unwrapped Phase")

            except Exception as e:
                print(f"Error rendering {p} in Quadrant {i+1}: {e}")
                traceback.print_exc()
                ax.text(0.5, 0.5, "Render Error", ha='center', color='red')

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = GatingApp()
    app.mainloop()