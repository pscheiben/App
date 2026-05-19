import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import traceback
import scipy.signal
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from network_engine import NetworkEngine

class GatingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OmniGate SI - Automated Hardware Analysis")        
        self.geometry("1600x950")
        
        self.engine = NetworkEngine()
        self.ref_engine = NetworkEngine() 
        
        self.raw_main_network = None
        self.raw_ref_network = None
        
        # Track filenames for dynamic title updates
        self.main_filename = None
        self.ref_filename = None
        
        self.plot_options = ["Magnitude (dB)", "Impulse Location", "Impedance (Ohm)", "Phase"]
        
        self.default_views = [
            ("S21", "Magnitude (dB)"),
            ("S11", "Magnitude (dB)"),
            ("S21", "Impulse Location"),
            ("S11", "Impedance (Ohm)")
        ]
        
        self._setup_ui()

    def _setup_ui(self):
        controls = ttk.Frame(self, padding="10")
        controls.pack(side=tk.LEFT, fill=tk.Y)

        btn_frame = ttk.Frame(controls)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Load Main File", command=self._load).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="Load Reference", command=self._load_ref).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        ttk.Label(controls, text="Port Remap (e.g., 2,1,4,3):", font=('', 9, 'italic')).pack(anchor="w", pady=(5, 0))
        map_frame = ttk.Frame(controls)
        map_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(map_frame, text="Main:").pack(side=tk.LEFT)
        self.ent_main_ports = ttk.Entry(map_frame, width=9)
        self.ent_main_ports.pack(side=tk.LEFT, padx=(2, 8))
        
        ttk.Label(map_frame, text="Ref:").pack(side=tk.LEFT)
        self.ent_ref_ports = ttk.Entry(map_frame, width=9)
        self.ent_ref_ports.pack(side=tk.LEFT, padx=(2, 0))
        
        ttk.Label(controls, text="Gate Center (ns):").pack(anchor="w", pady=(10, 0))
        self.ent_center = ttk.Entry(controls)
        self.ent_center.insert(0, "0.0")
        self.ent_center.pack(pady=2)

        ttk.Label(controls, text="Gate Span (ns):").pack(anchor="w")
        self.ent_span = ttk.Entry(controls)
        self.ent_span.insert(0, "2.0")
        self.ent_span.pack(pady=2)

        ttk.Label(controls, text="Reference Z0 (Ω):").pack(anchor="w", pady=(10, 0))
        self.ent_z0 = ttk.Entry(controls)
        self.ent_z0.insert(0, "50.0")
        self.ent_z0.pack(pady=2)

        ttk.Label(controls, text="Total Path Length (mm):").pack(anchor="w", pady=(10, 0))
        self.ent_dut_len = ttk.Entry(controls)
        self.ent_dut_len.insert(0, "140.0")
        self.ent_dut_len.pack(pady=2)

        self.lbl_er_calc = ttk.Label(controls, text="Calculated εr: --", font=('Arial', 10, 'bold'), foreground='#0052cc')
        self.lbl_er_calc.pack(anchor="w", pady=(5, 10))

        self.enable_gating = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Enable Gating", variable=self.enable_gating).pack(anchor="w", pady=5)

        self.quad_configs = []
        for i in range(4):
            frame = ttk.LabelFrame(controls, text=f"Quadrant {i+1}", padding="5")
            frame.pack(fill=tk.X, pady=5)
            
            p_sel = ttk.Combobox(frame)
            p_sel.pack(fill=tk.X)
            
            v_sel = ttk.Combobox(frame, values=self.plot_options)
            v_sel.pack(fill=tk.X)
            
            self.quad_configs.append({'param': p_sel, 'view': v_sel})

        ttk.Button(controls, text="Update Diagrams", command=self._update).pack(fill=tk.X, pady=20)
        
        ttk.Button(controls, text="Clear Reference Trace", command=self._clear_ref).pack(fill=tk.X, side=tk.BOTTOM)

        plot_container = ttk.Frame(self)
        plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        NavigationToolbar2Tk(self.canvas, plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_title(self):
        base_title = "OmniGate SI - Automated Hardware Analysis"
        if self.main_filename and self.ref_filename:
            self.title(f"{base_title}  -  [Main: {self.main_filename}  |  Ref: {self.ref_filename}]")
        elif self.main_filename:
            self.title(f"{base_title}  -  [Main: {self.main_filename}]")
        elif self.ref_filename:
            self.title(f"{base_title}  -  [Ref: {self.ref_filename}]")
        else:
            self.title(base_title)

    def _clear_ref(self):
        self.ref_engine.network = None
        self.raw_ref_network = None
        self.ent_ref_ports.delete(0, tk.END)
        self.ref_filename = None
        self._update_title()
        self._update()
        
    def _apply_port_mapping(self, raw_network, map_str):
        if not raw_network or not map_str.strip():
            return raw_network.copy() if raw_network else None
            
        try:
            new_order = [int(x.strip()) - 1 for x in map_str.split(',')]
            n = raw_network.nports
            if len(new_order) == n and set(new_order) == set(range(n)):
                nw = raw_network.copy()
                nw.s = nw.s[:, new_order, :][:, :, new_order]
                nw.z0 = nw.z0[:, new_order]
                return nw
            else:
                print(f"Warning: Port map must contain numbers 1 to {n} exactly once.")
        except Exception as e:
            print(f"Port mapping error: {e}")
            
        return raw_network.copy()

    def _load_ref(self):
        path = filedialog.askopenfilename(filetypes=[("Touchstone", "*.s2p *.s4p")])
        if path:
            self.ref_filename = os.path.basename(path)
            self._update_title()
            
            self.ref_engine.load_file(path)
            self.raw_ref_network = self.ref_engine.network.copy()
            self._update()

    def _load(self):
        path = filedialog.askopenfilename(filetypes=[("Touchstone", "*.s2p *.s4p")])
        if path:
            self.main_filename = os.path.basename(path)
            self._update_title()
            
            self.engine.load_file(path)
            self.raw_main_network = self.engine.network.copy()
            
            all_p = self.engine.get_available_params()
            
            for i, config in enumerate(self.quad_configs):
                config['param']['values'] = all_p
                target_p, target_v = self.default_views[i]
                
                if target_p in all_p:
                    config['param'].set(target_p)
                else:
                    config['param'].current(0)
                    
                config['view'].set(target_v)

            search_param = "S21" if "S21" in all_p else all_p[0]
            try:
                _, _, t_ns, _, y_imp = self.engine.get_processed_data(search_param, 0, 10, 50.0)
                
                mag_imp = np.abs(y_imp)
                idx_peak = np.argmax(mag_imp)
                arrival_time = t_ns[idx_peak]
                
                threshold = 0.10 * mag_imp[idx_peak]
                above_thresh = np.where(mag_imp > threshold)[0]
                
                if len(above_thresh) > 0:
                    pulse_indices = np.split(above_thresh, np.where(np.diff(above_thresh) != 1)[0] + 1)
                    target_pulse = [p for p in pulse_indices if idx_peak in p][0]
                    
                    t_start = t_ns[target_pulse[0]]
                    t_end = t_ns[target_pulse[-1]]
                    
                    detected_width = (t_end - t_start) * 1.5
                    final_span = max(1.5, min(detected_width, 8.0))
                else:
                    final_span = 2.0
                
                self.ent_center.delete(0, tk.END)
                self.ent_center.insert(0, f"{arrival_time:.3f}")
                
                self.ent_span.delete(0, tk.END)
                self.ent_span.insert(0, f"{final_span:.2f}")
                
            except Exception as e:
                print(f"Auto-detection failed: {e}")                
            
            self._update()

    def _update(self):
        if self.raw_main_network:
            self.engine.network = self._apply_port_mapping(self.raw_main_network, self.ent_main_ports.get())
            
        if self.raw_ref_network:
            self.ref_engine.network = self._apply_port_mapping(self.raw_ref_network, self.ent_ref_ports.get())
        
        if not self.engine.network: return
        
        try:
            c = float(self.ent_center.get())
            s = float(self.ent_span.get())
            z0 = float(self.ent_z0.get())
            dut_len = float(self.ent_dut_len.get())
        except ValueError:
            print("Input Error: Please ensure Gate, Z0, and Length values are numbers.")
            return

        self.fig.clear()
        self.lbl_er_calc.config(text="Calculated εr: --")
        
        gating_enabled = self.enable_gating.get()

        for i, config in enumerate(self.quad_configs):
            ax = self.fig.add_subplot(2, 2, i+1)
            p = config['param'].get()
            v = config['view'].get()
            
            if not p: continue

            has_ref = False
            if self.ref_engine.network and p in self.ref_engine.get_available_params():
                try:
                    ref_orig, ref_gated, ref_t_ns, ref_z_ohm, ref_y_imp = self.ref_engine.get_processed_data(p, c, s, z0)
                    has_ref = True
                except:
                    pass

            try:
                orig, gated, t_ns, z_ohm, y_imp = self.engine.get_processed_data(p, c, s, z0)
                ax.grid(True, alpha=0.3)

                if v == "Magnitude (dB)":
                    orig.plot_s_db(ax=ax, label='Main Raw', lw=1.5, color='#1f77b4')
                    if has_ref:
                        ref_orig.plot_s_db(ax=ax, label='Ref Raw', lw=1.2, color='darkred', linestyle='--')
                    
                    raw_ylim = ax.get_ylim()
                    
                    if gating_enabled:
                        gated.plot_s_db(ax=ax, label='Main Gated', lw=2.0, color='#ff7f0e')
                        if has_ref:
                            ref_gated.plot_s_db(ax=ax, label='Ref Gated', lw=1.5, color='red', linestyle='-.')
                            
                        ax.set_ylim(raw_ylim)
                        
                        mag = orig.s_db[:,0,0]
                        freqs = orig.frequency.f
                        valleys, _ = scipy.signal.find_peaks(-mag, prominence=1.5, distance=20)
                        
                        if len(valleys) >= 2:
                            idx1, idx2 = valleys[0], valleys[1]
                            f1, f2 = freqs[idx1], freqs[idx2]
                            delta_f_ghz = abs(f2 - f1) / 1e9
                            
                            ax.axvline(f1, color='purple', linestyle=':', alpha=0.7)
                            ax.axvline(f2, color='purple', linestyle=':', alpha=0.7)
                            
                            mid_f = (f1 + f2) / 2
                            y_pos = min(mag[idx1], mag[idx2]) - 5
                            
                            ax.text(mid_f, y_pos, f"Δf: {delta_f_ghz:.2f} GHz", 
                                    color='purple', fontsize=9, fontweight='bold', ha='center',
                                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
                            
                    ax.set_title(f"{p} Mag (Ref: {z0}Ω)")
                    ax.legend(loc='lower left', fontsize=8)
                    
                elif v == "Impulse Location":
                    orig.plot_s_db_time(ax=ax, lw=1.5, label='Main', color='#1f77b4')
                    if has_ref:
                        ref_orig.plot_s_db_time(ax=ax, lw=1.2, color='red', linestyle='--', label='Ref')
                    
                    if gating_enabled:
                        pos_mask = t_ns > 0
                        if np.any(pos_mask):
                            valid_t = t_ns[pos_mask]
                            valid_mag = np.abs(y_imp)[pos_mask]
                            idx_peak = np.argmax(valid_mag)
                            peak_time = valid_t[idx_peak]
                            
                            ax.axvline(peak_time, color='green', linestyle='--', alpha=0.8)
                            ax.text(peak_time + 0.3, -30, f"Arrival: {peak_time:.2f} ns", 
                                    color='green', fontsize=9, fontweight='bold', ha='left',
                                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
                            
                            if p[1] != p[2] and dut_len > 0: 
                                c_mm_ns = 299.792458
                                er_calc = ((peak_time * c_mm_ns) / dut_len) ** 2
                                self.lbl_er_calc.config(text=f"Calculated εr: {er_calc:.2f}")
                
                        ax.axvspan(c-s/2, c+s/2, color='orange', alpha=0.15)
                    
                    ax.set_title(f"{p} Impulse (Time of Flight)")
                    ax.legend(loc='upper right', fontsize=8)
                    
                elif v == "Impedance (Ohm)":
                    if any(x in p for x in ["11", "22", "33", "44"]):
                        ax.plot(t_ns, z_ohm, color='#1f77b4', lw=2.0, label='Main')
                        if has_ref:
                            ax.plot(ref_t_ns, ref_z_ohm, color='red', lw=1.5, linestyle='--', label='Ref')
                            
                        ax.set_ylabel("Impedance (Ω)", fontweight='bold')
                        ax.set_xlabel("Time (ns)")
                        
                        x_pad = max(s * 0.5, 0.5) 
                        x_min = c - (s / 2) - x_pad
                        x_max = c + (s / 2) + x_pad
                        ax.set_xlim(x_min, x_max)
                        
                        view_mask = (t_ns > x_min) & (t_ns < x_max)
                        active_z = z_ohm[view_mask] if np.any(view_mask) else z_ohm
                        z_min_val, z_max_val = np.min(active_z), np.max(active_z)
                        
                        if has_ref:
                            ref_active_z = ref_z_ohm[view_mask] if np.any(view_mask) else ref_z_ohm
                            z_min_val = min(z_min_val, np.min(ref_active_z))
                            z_max_val = max(z_max_val, np.max(ref_active_z))
                            
                        z_span = max(z_max_val - z_min_val, 4.0)
                        ax.set_ylim(z_min_val - (z_span * 0.15), z_max_val + (z_span * 0.15))
                        
                        if gating_enabled:
                            ax.axvspan(c-s/2, c+s/2, color='orange', alpha=0.12, lw=0)
                            
                            grad = np.gradient(z_ohm)
                            valid_view_idx = np.where(view_mask)[0] if np.any(view_mask) else np.arange(len(t_ns))
                            local_grad = grad[valid_view_idx]
                            
                            idx_fall = valid_view_idx[np.argmin(local_grad)]
                            t_start = t_ns[idx_fall]
                            
                            search_mask = (t_ns > (t_start + 0.05)) & (t_ns < x_max)
                            if np.any(search_mask):
                                valid_indices = np.where(search_mask)[0]
                                idx_rise = valid_indices[np.argmax(grad[search_mask])]
                            else:
                                idx_rise = idx_fall
                                
                            t_end = t_ns[idx_rise]
                            dip_width = t_end - t_start if t_end > t_start else 1e-9 
                                
                            ax.axvline(t_start, color='red', linestyle='--', alpha=0.6)
                            ax.axvline(t_end, color='red', linestyle='--', alpha=0.6)
                            
                            if t_end > t_start:
                                ax.axvspan(t_start, t_end, color='red', alpha=0.08)
                                ax.text(t_start + dip_width/2, z_min_val + (z_span * 0.05), f"Width: {dip_width:.2f} ns", 
                                        color='red', fontsize=9, fontweight='bold', ha='center',
                                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
                                
                                if dut_len > 0:
                                    ax_dist = ax.twiny()
                                    dist_min = ((x_min - t_start) / dip_width) * dut_len
                                    dist_max = ((x_max - t_start) / dip_width) * dut_len
                                    ax_dist.set_xlim(dist_min, dist_max)
                                    
                                    ax_dist.set_xlabel("Physical Distance into DUT (mm)", color='#0052cc', fontweight='bold')
                                    ax_dist.tick_params(axis='x', colors='#0052cc')

                                    trace_mask = (t_ns > (t_start + 0.05)) & (t_ns < (t_end - 0.05))
                                    if np.any(trace_mask):
                                        valid_t = t_ns[trace_mask]
                                        valid_z = z_ohm[trace_mask]
                                        
                                        peaks_ind, _ = scipy.signal.find_peaks(valid_z, prominence=0.4)
                                        peaks_cap, _ = scipy.signal.find_peaks(-valid_z, prominence=0.4)
                                        
                                        y_ruler = ax.get_ylim()[0] + (z_span * 0.02)
                                        
                                        for idx in peaks_ind:
                                            t_val = valid_t[idx]
                                            dist_val = ((t_val - t_start) / dip_width) * dut_len
                                            ax.plot([t_val, t_val], [ax.get_ylim()[0], y_ruler + (z_span*0.05)], color='darkorange', lw=2)
                                            ax.text(t_val, y_ruler + (z_span*0.06), f"{dist_val:.1f}", color='darkorange', fontsize=7, ha='center', rotation=90)
                                            
                                        for idx in peaks_cap:
                                            t_val = valid_t[idx]
                                            dist_val = ((t_val - t_start) / dip_width) * dut_len
                                            ax.plot([t_val, t_val], [ax.get_ylim()[0], y_ruler + (z_span*0.05)], color='darkred', lw=2)
                                            ax.text(t_val, y_ruler + (z_span*0.06), f"{dist_val:.1f}", color='darkred', fontsize=7, ha='center', rotation=90)
                                            
                        if has_ref:
                            ax.legend(loc='lower right', fontsize=8)
                            
                    else:
                        ax.text(0.5, 0.5, "Use Reflection for Z-Profile", ha='center', va='center')
                    ax.set_title(f"{p} Impedance Profile")
                    
                elif v == "Phase":
                    ax.plot(orig.frequency.f, np.unwrap(np.angle(orig.s[:,0,0], deg=True)), lw=1.5, label='Main', color='#1f77b4')
                    if has_ref:
                        ax.plot(ref_orig.frequency.f, np.unwrap(np.angle(ref_orig.s[:,0,0], deg=True)), lw=1.2, color='red', linestyle='--', label='Ref')
                        ax.legend(loc='best', fontsize=8)
                        
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