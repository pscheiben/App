import skrf as rf
import numpy as np

class NetworkEngine:
    def __init__(self):
        self.network = None

    def load_file(self, file_path):
        nw = rf.Network(file_path)
        # Extrapolate to DC for accurate TDR/Step response
        self.network = nw.extrapolate_to_dc()
        return self.network

    def get_available_params(self):
        if not self.network: return []
        n = self.network.nports
        return [f"S{i+1}{j+1}" for i in range(n) for j in range(n)]

    def get_processed_data(self, param_str, center, span, z0_ref=50.0):
        if not self.network: return None, None, None, None, None
        
        # Parse S-parameter string (e.g., "S21")
        i, j = int(param_str[1])-1, int(param_str[2])-1
        one_port = rf.Network(frequency=self.network.frequency, 
                             s=self.network.s[:, i, j], 
                             z0=self.network.z0[:, i])
        
        # Normalize to custom reference
        one_port.z0 = np.full(len(one_port), z0_ref)
        
        # Frequency domain gating
        gated = one_port.time_gate(center=center, span=span, t_unit='ns')
        
        # TDR/TDT Impulse
        t, y_imp = one_port.impulse_response(pad=1000)
        t_ns = t * 1e9
        
        # Impedance Step Response
        t_step, y_step = one_port.step_response(pad=1000)
        z_t = z0_ref * (1 + np.real(y_step)) / (1 - np.real(y_step) + 1e-9)
            
        return one_port, gated, t_ns, z_t, y_imp