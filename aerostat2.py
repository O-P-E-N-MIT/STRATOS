import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import brentq, minimize
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from enum import Enum
import csv

class MaterialClass(Enum):
    STANDARD = "Standard"
    HIGH_TEMP = "High temperature"
    COLD_TEMP = "Cold temperatue"
    EXTREME = "Extreme environemnt"

MATERIAL_MAP = {
    "Standard": MaterialClass.STANDARD,
    "High temperature": MaterialClass.HIGH_TEMP,
    "Cold temperature": MaterialClass.COLD_TEMP,
    "Extreme environment": MaterialClass.EXTREME,
}

@dataclass
class ThermalStressProperties:
    cte: float
    max_temp: float
    min_temp: float
    base_strength: float
    temp_derating: float
    fatigue_factor: float
    uv_degradation: float

class ThermalEnvelopeSystem:
    def __init__(self):
        self.material_props = {
            MaterialClass.STANDARD: ThermalStressProperties(
                cte=2.3e-5, max_temp=323.15, min_temp=233.15,
                base_strength=75.0, temp_derating=0.15,
                fatigue_factor=0.995, uv_degradation=0.02
            ),
            MaterialClass.HIGH_TEMP: ThermalStressProperties(
                cte=1.8e-5, max_temp=373.15, min_temp=233.15,
                base_strength=95.0, temp_derating=0.08,
                fatigue_factor=0.998, uv_degradation=0.015
            ),
            MaterialClass.COLD_TEMP: ThermalStressProperties(
                cte=1.5e-5, max_temp=333.15, min_temp=213.15,
                base_strength=85.0, temp_derating=0.12,
                fatigue_factor=0.997, uv_degradation=0.01
            ),
            MaterialClass.EXTREME: ThermalStressProperties(
                cte=1.2e-5, max_temp=393.15, min_temp=193.15,
                base_strength=110.0, temp_derating=0.05,
                fatigue_factor=0.999, uv_degradation=0.008
            )
        }

    def calculate_thermal_stress(self, envelope_temp, ambient_temp,material_class, pressure_diff_pa, radius_m, thickness_m):
        props = self.material_props[material_class]

        # Thermal part
        delta_T = envelope_temp - ambient_temp
        thermal_strain = props.cte * delta_T  # dimensionless
        thermal_stress = thermal_strain * props.base_strength

        # Hoop stress from pressure (thin shell, approximate spherical/prolate)
        # σ = Δp * r / (2 t) [Pa] → convert to MPa
        if thickness_m <= 0:
            pressure_stress = 0.0
        else:
            sigma_pa = pressure_diff_pa * radius_m / (2.0 * thickness_m)
            pressure_stress = sigma_pa / 1e6  # Pa → MPa

        total_stress = thermal_stress + pressure_stress

        # Temperature derating on material strength
        if envelope_temp > 293.15:
            derating = 1.0 - (envelope_temp - 293.15) * props.temp_derating / 100.0
            derating = max(0.0, derating)
            total_stress *= derating
        else:
            derating = 1.0

        margin = props.base_strength - total_stress

        return {"thermal_stress": thermal_stress,
            "pressure_stress": pressure_stress,
            "total_stress": total_stress,
            "margin": margin,
            "derating_factor": derating,
            "cte": props.cte}

    def envelope_thermal_model(self, altitude, solar_flux=1000,
                               emissivity=0.8, absorptivity=0.3,
                               wind_speed=5):
        P, T_ambient = self.stdatmo(altitude)
        h_conv = 10.45 - wind_speed + 10.0 * np.sqrt(wind_speed)
        q_solar = solar_flux * absorptivity
        q_ir = emissivity * 5.67e-8 * (T_ambient ** 4)
        T_env = T_ambient + (q_solar - q_ir) / max(h_conv, 1e-3)
        return T_env, T_ambient

    def stdatmo(self, h):
        T0, P0, L, R, g = 288.15, 101325.0, 0.0065, 287.0, 9.80665
        if h < 11000.0:
            T = T0 - L * h
            P = P0 * (T / T0) ** (g / (R * L))
        else:
            T = 216.65
            P = 22632.0 * np.exp(-g * (h - 11000.0) / (R * T))
        return P, T
    
class StressAwareAerostat:
    def __init__(self, base_geometry, thermal_system):
        self.geometry = base_geometry  # AerostatGeometry
        self.thermal = thermal_system
        self.stress_history = []

    def _estimate_thickness(self, sig_env_kg_m2, material_density_kg_m3=1400.0):
        return sig_env_kg_m2 / material_density_kg_m3

    def calculate_burst_alt(self, V, material_class,sig_env_kg_m2=0.75, safety_factor=2.0):
        L, D = self.geometry.get_dims(V)
        radius = D / 2.0
        t_env = self._estimate_thickness(sig_env_kg_m2)

        props = self.thermal.material_props[material_class]
        allowable_stress = props.base_strength / safety_factor  # MPa

        def stress_margin(h):
            P, T_amb = self.thermal.stdatmo(h)
            T_env, _ = self.thermal.envelope_thermal_model(h)
            
            delta_p = 500.0  # Pa
            res = self.thermal.calculate_thermal_stress(
                T_env, T_amb, material_class,
                pressure_diff_pa=delta_p,
                radius_m=radius,
                thickness_m=t_env,
            )
            return allowable_stress - res["total_stress"]

        try:
            h_burst = brentq(stress_margin, 0.0, 20000.0)
            return h_burst
        except ValueError:
            # No root in range; treat as “no burst below 20 km”
            return 20000.0

    def operational_envelope(self, V, material_class):
        h_range = np.linspace(0, 10000.0, 50)
        boundaries = {"safe": [], "warning": [], "critical": []}
        for h in h_range:
            T_env, _ = self.thermal.envelope_thermal_model(h)
            props = self.thermal.material_props[material_class]
            if T_env < props.min_temp or T_env > props.max_temp:
                category = "critical"
            elif (T_env < props.min_temp + 20.0 or
                  T_env > props.max_temp - 20.0):
                category = "warning"
            else:
                category = "safe"
            boundaries[category].append(h)
        return boundaries

class GNVRGeometry:
    def vol_from_dim(self, length, diameter):
        a = length / 2
        b = diameter / 2
        return (4/3) * np.pi * a * b**2
    
    def dim_from_vol(self, V, Fineness_ratio=1.5):
        b = (V * 3/ (4 * np.pi * Fineness_ratio))**(1/3)
        a = Fineness_ratio * b
        return 2*a, 2*b
    
    def surface_area(self, length, diameter):
        a = length / 2
        b = diameter / 2
        e = np.sqrt(1 - (b**2/a**2)) if a > b else np.sqrt(1 - (a**2/b**2))
        if a == b:
            return 4 * np.pi * a**2
        elif a > b:
            return 2 * np.pi * b**2 * (1 + (a/(b*e)) * np.arcsin(e))
        else:
            return 2 * np.pi * a**2 * (1 + ((1-e**2)/e) * np.arctanh(e))
        
    def drag_coef(self, fineness_ratio):
        if fineness_ratio <= 1.2:
            return 0.47
        elif fineness_ratio <= 2.0:
            return 0.35
        elif fineness_ratio <= 3.0:
            return 0.20
        else:
            return 0.15
        
class LYNXGeometry:
    def vol_from_dim(self, length, max_dia):
        R = max_dia / 2
        return (np.pi * length * R**2) / 4
    
    def dim_from_vol(self, V, fineness_ratio=3.0):
        R = np.cbrt((4*V) / (np.pi * fineness_ratio))
        length = fineness_ratio * 2 * R
        return length, 2*R
    
    def surface_area(self, length, max_dia):
        a = length / 2
        b = max_dia / 2
        e = np.sqrt(1 - (b**2 / a**2))
        return 2 * np.pi * b**2 * (1 + (a/(b*e)) * np.arcsin(e))
    
    @staticmethod
    def profile_coordiantes(length, max_dia, n_points=200):
        x = np.linspace(0, length, n_points)
        c = length
        t = 0.18
        yt = 5*t*c*(0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1015*(x/c)**4)
        scale = max_dia / (2 * np.max(yt))
        y_upper = yt * scale
        y_lower = -yt * scale
        return x, y_upper, y_lower
    
    def drag_coef(self, fineness_ratio, Re=1e6):
        cf = 0.074 / (Re**0.2)
        cd_form = 0.002 * (fineness_ratio - 3.0)**2
        return cf * (1 + 1.5/(fineness_ratio**1.5)) + cd_form

class AerostatGeometry:
    def __init__(self, shape, fineness_ratio=1.5):
        self.shape = shape
        self.fineness_ratio = fineness_ratio

        if shape == "GNVR":
            self.geometry = GNVRGeometry()
        elif shape == "LYNX":
            self.geometry = LYNXGeometry()
        else:
            raise ValueError("Unknown shape", shape)
        
    def get_dims(self, V):
        return self.geometry.dim_from_vol(V, self.fineness_ratio)
    
    def get_surface_area(self, V):
        L, D = self.get_dims(V)
        return self.geometry.surface_area(L, D)
    
    def get_drag_coef(self, V, wind_speed=10):
        L, D = self.get_dims(V)
        Re = wind_speed * L / 1.5e-5
        return self.geometry.drag_coef(self.fineness_ratio, Re) if self.shape == "LYNX" else self.geometry.drag_coef(self.fineness_ratio)
    
    def get_stress_factors(self, V):
        if self.shape == "GNVR":
            return (1.0, 2.0)
        else:
            return (1.2, 1.5)

class FinalAerostatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aerostat Pro: Final Design & Ascent Dynamics")
        self.root.geometry("1600x1000")
        self.fig_frame_main = None
        self.fig_frame_stress = None

        self.thermal_system = ThermalEnvelopeSystem()
        
        # Physics Constants (MATLAB Source)
        self.g, self.RDwv = 9.80665, 0.622
        self.rho0, self.P0, self.T0 = 1.225, 101325, 288.15
        self.K = self.rho0 * self.g * self.T0 / self.P0
        self.DeltaP, self.DeltaTsh = 500, 5
        self.Cd = 0.45 
        self.rh = 0.7
        self.purity = 0.97
        self.geometry = None
        
        self.V_env_last = None
        self.L_last = None
        self.D_last = None

        self.setup_ui()

        self.thermal_system.stdatmo = self.stdatmo

    def stdatmo(self, h):
        T0, P0, L, R, g = 288.15, 101325, 0.0065, 287, 9.80665
        if h < 11000:
            T = T0 - L * h
            P = P0 * (T / T0)**(g / (R * L))
        else:
            T = 216.65
            P = 22632 * np.exp(-g * (h - 11000) / (R * T))
        return P, T

    def vapor_pressure(self, T):
        Tc = T - 273.15
        return self.rh * 611.21 * np.exp((18.678 - Tc/234.5)*(Tc/(257.14 + Tc)))
    
    def perform_stress_analysis(self, V, material_class):
        stress_aware = StressAwareAerostat(self.geometry, self.thermal_system)
        h_burst = stress_aware.calculate_burst_alt(V, material_class)
        boundaries = stress_aware.operational_envelope(V, material_class)
        h_op = self.p_h_oper.get()
        T_env, T_amb = self.thermal_system.envelope_thermal_model(h_op)
        stress_results = self.thermal_system.calculate_thermal_stress(T_env, T_amb, material_class, self.DeltaP)

        self.out_box.insert(tk.END, f"\n\n--- STRESS ANALYSIS ---\n")
        self.out_box.insert(tk.END, f"Material Class: {material_class.value}\n")
        self.out_box.insert(tk.END, f"Envelope Temp: {T_env-273.15:.1f}°C\n")
        self.out_box.insert(tk.END, f"Ambient Temp: {T_amb-273.15:.1f}°C\n")
        self.out_box.insert(tk.END, f"Thermal Stress: {stress_results['thermal_stress']:.1f} MPa\n")
        self.out_box.insert(tk.END, f"Pressure Stress: {stress_results['pressure_stress']:.1f} MPa\n")
        self.out_box.insert(tk.END, f"Total Stress: {stress_results['total_stress']:.1f} MPa\n")
        self.out_box.insert(tk.END, f"Safety Margin: {stress_results['margin']:.1f} MPa\n")
        self.out_box.insert(tk.END, f"Burst Altitude: {h_burst:.0f} m\n")

        props = self.thermal_system.material_props[material_class]
        self.out_box.insert(tk.END, f"Temp Range: {props.min_temp-273.15:.0f}C to {props.max_temp-273.15:.0f}C\n" )
        self.plot_stress_analysis(V, material_class, stress_results, boundaries)
        
    def calculate_with_stress_analysis(self):
        try:
            material_label = self.p_material_class.get()
            material_class = MATERIAL_MAP[material_label]

            self.calculate()

            if self.geometry and self.V_env_last is not None:
                sig_env = self.p_sig_env.get()
                stress_aware = StressAwareAerostat(self.geometry, self.thermal_system)
                h_burst = stress_aware.calculate_burst_alt(self.V_env_last, material_class, sig_env_kg_m2=sig_env,
                    safety_factor=self.p_safety_factor.get())
                boundaries = stress_aware.operational_envelope(self.V_env_last, material_class)

                h_op = self.p_h_oper.get()
                T_env, T_amb = self.thermal_system.envelope_thermal_model(h_op)
                radius = self.D_last / 2.0
                # thickness from areal density
                t_env = stress_aware._estimate_thickness(sig_env)

                stress_results = self.thermal_system.calculate_thermal_stress(T_env, T_amb, material_class,
                    pressure_diff_pa=self.DeltaP,
                    radius_m=radius,
                    thickness_m=t_env)

                self.out_box.insert(tk.END, "\n\n--- STRESS ANALYSIS ---\n")
                self.out_box.insert(tk.END, f"Material Class: {material_class.value}\n")
                self.out_box.insert(tk.END, f"Envelope Temp: {T_env-273.15:.1f}°C\n")
                self.out_box.insert(tk.END, f"Ambient Temp: {T_amb-273.15:.1f}°C\n")
                self.out_box.insert(tk.END, f"Thermal Stress: {stress_results['thermal_stress']:.2f} MPa\n")
                self.out_box.insert(tk.END, f"Pressure Stress: {stress_results['pressure_stress']:.2f} MPa\n")
                self.out_box.insert(tk.END, f"Total Stress: {stress_results['total_stress']:.2f} MPa\n")
                self.out_box.insert(tk.END, f"Safety Margin: {stress_results['margin']:.2f} MPa\n")
                self.out_box.insert(tk.END, f"Burst Altitude: {h_burst:.0f} m\n")

                props = self.thermal_system.material_props[material_class]
                self.out_box.insert(tk.END, f"Temp Range: {props.min_temp-273.15:.0f}C to {props.max_temp-273.15:.0f}C\n")

                self.plot_stress_analysis(self.V_env_last, material_class, stress_results, boundaries)
        except Exception as e:
            messagebox.showerror("Stress Analysis Error", str(e))

    
    def net_lift(self, V, h, I, m_fixed, m_env, m_fin, m_ball_fab):
        P, T = self.stdatmo(h)
    
        # Air properties
        e = self.vapor_pressure(T)
        rho_air = P / (287 * T)
        rho_moist = (self.T0 / self.P0) * self.rho0 * (P - (1 - self.RDwv) * e ) / T
    
        # Lifting gas properties
        gas = self.p_gas.get()
        rg = 2.016 / 28.964 if gas == "Hydrogen" else 4.003 / 28.964
        eff_rg = 1 - (1 - rg) * self.purity
        rho_lg = eff_rg * (P + self.DeltaP) / (287 * (T + self.DeltaTsh))
        m_lg = rho_lg * I * V
    
        # Ballonet air volume (30% of envelope when empty)
        Vb_max = 0.3 * V
        Vb = (1 - I) * V
        Vb = min(Vb, Vb_max)
        m_ba = rho_air * Vb
    
        # Buoyancy
        vol = I * V + Vb
        Lg = rho_moist * self.g * vol
    
        # Total mass
        total_mass = m_lg + m_ba + m_fixed + m_env + m_fin + m_ball_fab
    
        return Lg - total_mass * self.g

    def calculate(self):
        try:
            # Inputs from GUI
            m_pay = self.p_pay.get()
            m_struct = self.p_struct.get()
            h_dep = self.p_h_dep.get()
            h_oper = self.p_h_oper.get()
            h_margin = self.p_h_marg.get()
            h_pressure = h_oper + h_margin
            rh = self.p_rh.get()
            self.rh = rh
            roc = self.p_roc.get()
            shape = self.p_shape.get()
            fineness = float(self.p_fineness.get())
            self.geometry = AerostatGeometry(shape, fineness)
            self.Cd = self.geometry.get_drag_coef(10000)
            
            tether_len = h_oper * 1.25 
            m_tether = tether_len * self.p_tether_rho.get()
            m_fixed = m_pay + m_struct + m_tether
            
            # Material Densities 
            sig_env, sig_bal, sig_fin = self.p_sig_env.get(), self.p_sig_bal.get(), self.p_sig_fin.get()
            
            gas = self.p_gas.get()
            rg = 2.016 / 28.964 if gas == "Hydrogen" else 4.003 / 28.964
            
            p_dep, t_dep = self.stdatmo(h_dep)
            p_op, t_op = self.stdatmo(h_oper)
            p_pressure, t_pressure = self.stdatmo(h_pressure)

            # Sizing Solver
            def root_v(V):
                L, D = self.geometry.get_dims(V)
                sa_env = self.geometry.get_surface_area(V)
                m_env = sig_env * sa_env
                Vb_ref = V * 0.3
                r_b = (Vb_ref / np.pi)**(1/3)
                m_ball_fab = sig_bal * (3 * np.pi * r_b**2)
            
                # Fin mass (15% of envelope surface area)
                fin_ratio = 0.15 if shape == "GNVR" else 0.1
                m_fin = sig_fin * (sa_env * fin_ratio)
            
                # Net lift at pressure altitude: full inflation, no ballonet air
                return self.net_lift(V, h_pressure, 1.0, m_fixed, m_env, m_fin, m_ball_fab)

            V_low, V_high = 500.0, 500000.0
            f_low = root_v(V_low)
            f_high = root_v(V_high)

            if f_low * f_high > 0:
                messagebox.showerror("Design Error", "No feasible envelope volume for given inputs.\n"
                                     "Try reducing payload/structure mass, increasing fabric strength, "
                                     "or increasing allowed fin/ballonet fractions."
                )
                return

            V_env = brentq(root_v, V_low, V_high)

            
            # --- Results Preparation ---
            L, D = self.geometry.get_dims(V_env)
            sa_env = self.geometry.get_surface_area(V_env)
            m_env = sig_env * sa_env
            Vb_ref = V_env * 0.3
            r_b = (Vb_ref / np.pi)**(1/3)
            m_ball_fab = sig_bal * (3 * np.pi * r_b**2)
            fin_ratio = 0.15 if shape == "GNVR" else 0.1
            m_fin = sig_fin * (sa_env * fin_ratio)
            self.V_env_last = V_env
            self.L_last = L
            self.D_last = D

            def root_I(I):
                return self.net_lift(V_env, h_oper, I, m_fixed, m_env, m_fin, m_ball_fab)
        
            I_low, I_high = 0.0, 1.0
            fI_low = root_I(I_low)
            fI_high = root_I(I_high)

            if fI_low * fI_high > 0:
                messagebox.showerror("Design Error",
                    "No feasible inflation fraction at operating altitude.\n"
                    "Try reducing total mass (payload/structure/tether) or "
                    "increasing envelope volume."
                )
                return

            I_oper = brentq(root_I, I_low, I_high)

            
            # Compute m_lg using consistent formula
            eff_rg = 1 - (1 - rg) * self.purity
            rho_lg_op = eff_rg * (p_op + self.DeltaP) / (287 * (t_op + self.DeltaTsh))
            m_lg = rho_lg_op * I_oper * V_env
            
            # Simulated Profiles
            h_vec = np.linspace(h_dep, h_pressure, 120)
            lift, inf, vb, t_vec, flow = [], [], [], [], []
            
            prev_vb = None
            prev_h = None
            for h in h_vec:
                P, T = self.stdatmo(h)
                rho_lg = eff_rg * (P + self.DeltaP) / (287 * (T + self.DeltaTsh))
                gas_vol = m_lg / rho_lg
                if gas_vol > V_env:
                    gas_vol = V_env
                Vb_max = 0.3 * V_env
                curr_vb = max(0, min(V_env - gas_vol, Vb_max))
                I = gas_vol / V_env
                
                if prev_h is None:
                    time_step = 0
                else:
                    time_step = (h - prev_h) / roc
                discharge = 0 if prev_vb is None else (prev_vb - curr_vb) / time_step
                
                inf.append(I)
                vb.append(curr_vb)
                lift.append(self.net_lift(V_env, h, I, m_fixed, m_env, m_fin, m_ball_fab))
                if prev_h is None:
                    t_vec.append(0)
                else:
                    t_vec.append(t_vec[-1] + time_step)
                flow.append(discharge)
                prev_vb = curr_vb
                prev_h = h

            res = {'h': h_vec, 't': t_vec, 'flow': flow}
            m_lg_display = m_lg
            self.plot_all(h_vec, lift, inf, vb, V_env, m_env, m_fin, m_ball_fab, m_lg_display, res)
            
        except Exception as e:
            messagebox.showerror("Calculation Error", str(e))

    def setup_ui(self):
        container = ttk.Frame(self.root, padding="10")
        container.pack(fill=tk.BOTH, expand=True)

        # Left side: notebook with input tabs
        left_panel = ttk.Frame(container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        req_frame = ttk.Frame(notebook)
        env_frame = ttk.Frame(notebook)
        stress_frame = ttk.Frame(notebook)

        notebook.add(req_frame, text="Sizing & Payload")
        notebook.add(env_frame, text="Environment")
        notebook.add(stress_frame, text="Stress & Materials")

        #right notebook
        right_notebook = ttk.Notebook(container)
        right_notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        main_tab = ttk.Frame(right_notebook)
        stress_tab = ttk.Frame(right_notebook)

        right_notebook.add(main_tab, text="Ascent & Shape")
        right_notebook.add(stress_tab, text="Stress & Thermal")

        self.fig_frame_main = ttk.Frame(main_tab)
        self.fig_frame_main.pack(fill=tk.BOTH, expand=True)

        self.fig_frame_stress = ttk.Frame(stress_tab)
        self.fig_frame_stress.pack(fill=tk.BOTH, expand=True)

        # Below notebook: buttons + output
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        
        # New Requirement Inputs
        self.p_pay = tk.DoubleVar(value=250.0)
        self.p_struct = tk.DoubleVar(value=150.0)
        self.p_tether_rho = tk.DoubleVar(value=0.15)
        self.p_sig_env = tk.DoubleVar(value=0.75)
        self.p_sig_bal = tk.DoubleVar(value=0.35)
        self.p_sig_fin = tk.DoubleVar(value=0.50)
        self.p_h_dep = tk.DoubleVar(value=100)
        self.p_h_oper = tk.DoubleVar(value=4500.0)
        self.p_h_marg = tk.DoubleVar(value=500)
        self.p_rh = tk.DoubleVar(value=0.7)
        self.p_roc = tk.DoubleVar(value=1.5)
        self.p_gas = tk.StringVar(value="Hydrogen")
        self.p_shape = tk.StringVar(value="GNVR")
        self.p_fineness = tk.StringVar(value=1.5)

        self.p_material_class = tk.StringVar(value="Standard")
        self.p_solar_flux = tk.DoubleVar(value=1000)
        self.p_wind_speed = tk.DoubleVar(value = 5)
        self.p_emissivity = tk.DoubleVar(value=0.8)
        self.p_absorptivity = tk.DoubleVar(value=0.3)
        self.p_design_lifetime = tk.IntVar(value = 5)

        rows = [("Payload (kg)", self.p_pay), ("Structure (kg)", self.p_struct),
                ("Tether (kg/m)", self.p_tether_rho), ("Env Fab (kg/m2)", self.p_sig_env),
                ("Ballonet Fab", self.p_sig_bal), ("Fin Fab", self.p_sig_fin),
                ("Op Alt (m)", self.p_h_oper), ("dep Alt (m)", self.p_h_dep),
                ("margin Alt (m)", self.p_h_marg), ("RH (0-1)", self.p_rh), ("ROC (m/s)", self.p_roc),
                ("Fineness Ratio", self.p_fineness)]
        
        for i, (l, v) in enumerate(rows):
            ttk.Label(req_frame, text=l).grid(row=i, column=0, sticky="w")
            ttk.Entry(req_frame, textvariable=v, width=10).grid(row=i, column=1)
        
        ttk.Label(req_frame, text="Lifting Gas").grid(row=5, column=0, pady=5)
        ttk.Combobox(req_frame, textvariable=self.p_gas, values=["Hydrogen", "Helium"]).grid(row=5, column=1)
        ttk.Label(req_frame, text="Envelope Shape").grid(row=6, column=0, pady=5)
        ttk.Combobox(req_frame, textvariable=self.p_shape, values=["GNVR", "LYNX"], width=10).grid(row=6, column=1)

        self.out_box = tk.Text(left_panel, height=8, width=30, font=("Consolas", 9)); self.out_box.pack(fill=tk.BOTH, expand=False, pady=5)
        self.fig_frame = ttk.Frame(container); self.fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.p_fabric = tk.StringVar(value="polyester")
        ttk.Label(stress_frame, text="Material Class").grid(row=0, column=0, sticky="w")
        ttk.Combobox(stress_frame, textvariable=self.p_material_class, values=list(MATERIAL_MAP.keys()), width=14).grid(row=0, column=1)

        ttk.Label(stress_frame, text="Fabric Material").grid(row=1, column=0, sticky="w")
        ttk.Combobox(stress_frame, textvariable=self.p_fabric, values=["polyester", "polyurethane"], width=14).grid(row=1, column=1)
        
        #temperature range
        ttk.Label(stress_frame, text="Min Temp (C)").grid(row=1,column=0, sticky="w")
        ttk.Label(stress_frame, text="Max Temp (C)").grid(row=2, column=0, sticky="w")
        ttk.Label(stress_frame, text="Safety Factor").grid(row=3, column=0, sticky="w")
        self.p_temp_min = tk.DoubleVar(value=-40)
        self.p_temp_max = tk.DoubleVar(value=50)
        self.p_safety_factor = tk.DoubleVar(value=4.0)
        ttk.Entry(stress_frame, textvariable=self.p_temp_min, width=10).grid(row=2, column=1)
        ttk.Entry(stress_frame, textvariable=self.p_temp_max, width=10).grid(row=3, column=1)
        ttk.Entry(stress_frame, textvariable=self.p_safety_factor, width=10).grid(row=4, column=1)
        
        env_params = [("Solar Flux (W/m²)", self.p_solar_flux), ("Wind Speed (m/s)", self.p_wind_speed),
            ("Emissivity", self.p_emissivity), ("Absorptivity", self.p_absorptivity),
            ("Design Life (years)", self.p_design_lifetime)]
        
        for i, (l, v) in enumerate(env_params):
            ttk.Label(env_frame, text=l).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(env_frame, textvariable=v, width=12).grid(row=i, column=1, pady=2)
        
        ttk.Button(btn_frame, text="Standard Analysis", command=self.calculate).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stress Analysis", command=self.calculate_with_stress_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Export LYNX Profile", command=self.export_lynx_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Export GNVR Profile", command=self.export_gnvr_profile).pack(side=tk.LEFT, padx=2)


    def plot_shape_profile(self, ax, V):
        L, D = self.geometry.get_dims(V)
        
        if self.p_shape.get() == "LYNX":
            x, y_upper, _ = LYNXGeometry.profile_coordiantes(L, D)
            theta = np.linspace(0, 2*np.pi, 50)
            X, Theta = np.meshgrid(x, theta)
            R = np.tile(y_upper, (len(theta), 1))
            Y = R * np.cos(Theta)
            Z = R * np.sin(Theta)
            ax.plot_surface(X, Y, Z, alpha=0.7, color='skyblue', edgecolor='none')
        else:
            # GNVR as ellipsoid
            a, b = L/2, D/2
            u = np.linspace(0, 2*np.pi, 40)
            v = np.linspace(0, np.pi, 40)
            x = a * np.outer(np.cos(u), np.sin(v))
            y = b * np.outer(np.sin(u), np.sin(v))
            z = b * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(x, y, z, alpha=0.7, color='skyblue', edgecolor='k', linewidth=0.1)
        
        max_dim = max(L, D)
        ax.set_xlim(-max_dim/2, max_dim/2)
        ax.set_ylim(-max_dim/2, max_dim/2)
        ax.set_zlim(-max_dim/2, max_dim/2)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
        ax.set_title(f'{self.p_shape.get()} 3D Shape\nL={L:.1f}m, D={D:.1f}m')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.grid(True)
    
    def plot_shape_profile_2d(self, ax, V):
        L, D = self.geometry.get_dims(V)

        if self.p_shape.get() == "LYNX":
            x, y_upper, y_lower = LYNXGeometry.profile_coordiantes(L, D)
            ax.plot(x, y_upper, 'b')
            ax.plot(x, y_lower, 'b')
            ax.set_title("LYNX Side Profile")
        else:
            # GNVR as 2D ellipse in x–z plane
            a = L / 2.0
            b = D / 2.0
            t = np.linspace(0.0, 2.0 * np.pi, 400)
            x = a * np.cos(t)
            z = b * np.sin(t)
            ax.plot(x, z, 'b')
            ax.set_title("GNVR Side Profile")

        ax.set_aspect('equal', 'box')
        ax.set_xlabel("Length axis (m)")
        ax.set_ylabel("Diameter axis (m)")
        ax.grid(True)

    import csv

    def export_lynx_profile(self, filename="lynx_profile.csv"):
        if self.geometry is None or self.V_env_last is None:
            messagebox.showerror("Export Error", "Run sizing first.")
            return

        # Ensure shape is LYNX
        L, D = self.geometry.get_dims(self.V_env_last)
        x, y_upper, y_lower = LYNXGeometry.profile_coordiantes(L, D)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_m", "y_upper_m", "y_lower_m"])
            for xi, yu, yl in zip(x, y_upper, y_lower):
                writer.writerow([xi, yu, yl])

        messagebox.showinfo("Export", f"Saved LYNX profile to {filename}")

    def export_gnvr_profile(self, filename="gnvr_profile.csv", n_points=400):
        if self.geometry is None or self.V_env_last is None:
            messagebox.showerror("Export Error", "Run sizing first.")
            return

        L, D = self.geometry.get_dims(self.V_env_last)
        a = L / 2.0
        b = D / 2.0

        t = np.linspace(0.0, 2.0 * np.pi, n_points)
        x = a * np.cos(t)
        z = b * np.sin(t)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_m", "z_m"])
            for xi, zi in zip(x, z):
                writer.writerow([xi, zi])

        messagebox.showinfo("Export", f"Saved GNVR profile to {filename}")


    def plot_stress_analysis(self, V, material_class, stress_results, boundaries):

        for w in self.fig_frame_stress.winfo_children():
            w.destroy()

        fig_stress, axes = plt.subplots(2, 2, figsize=(10,8))
        h_range = np.linspace(0, 10000, 100)
        stress_aware = StressAwareAerostat(self.geometry, self.thermal_system)
        T_env_range = []
        T_amb_range = []
        stress_range = []

        for h in h_range:
            T_env, T_amb = self.thermal_system.envelope_thermal_model(h)
            T_env_range.append(T_env + 273.15)
            T_amb_range.append(T_amb + 273.15)
            res = self.thermal_system.calculate_thermal_stress(
            T_env, T_amb, material_class,
            pressure_diff_pa=self.DeltaP,
            radius_m=self.D_last / 2.0,
        thickness_m= stress_aware._estimate_thickness(self.p_sig_env.get()))
        stress_range.append(res["total_stress"])

        axes[0,0].plot(h_range, T_env_range, 'r-', label='Envelope')
        axes[0,0].plot(h_range, T_amb_range, 'b--', label='Ambient')
        axes[0,0].axhline(y=stress_results['derating_factor']*100, color='g', linestyle=':', label='Strength Derating %')
        axes[0,0].set_xlabel('Altitude (m)')
        axes[0,0].set_ylabel('Temperature (C)')
        axes[0,0].set_title('Thermal Profile')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        stress_range = []
        for h in h_range:
            T_env, T_amb = self.thermal_system.envelope_thermal_model(h)
            res = self.thermal_system.calculate_thermal_stress(T_env, T_amb, material_class, self.DeltaP, radius_m=self.D_last / 2.0,
                                                               thickness_m= stress_aware._estimate_thickness(self.p_sig_env.get()))
            stress_range.append(res['total_stress'])

        axes[0,1].plot(h_range, stress_range, 'purple')
        props = self.thermal_system.material_props[material_class]
        axes[0,1].axhline(y=props.base_strength, color='r', linestyle='--', label='Material limit')
        axes[0,1].axhline(y=props.base_strength/2, color='orange', 
                         linestyle=':', label='Design Limit')
        axes[0,1].set_xlabel('Altitude (m)')
        axes[0,1].set_ylabel('Stress (MPa)')
        axes[0,1].set_title('Structural Stress Profile')
        axes[0,1].legend()
        axes[0,1].grid(True)

        axes[1,0].set_title('Operational Temperature Envelope')
        axes[1,0].set_xlabel('Altitude (m)')
        axes[1,0].set_ylabel('Temperature (°C)')

        h_plot = []
        temp_plot = []
        for h in h_range:
            T_env, _ = self.thermal_system.envelope_thermal_model(h)
            h_plot.append(h)
            temp_plot.append(T_env - 273.15)
        
        axes[1,0].plot(h_plot, temp_plot, 'b-', linewidth=2)
        axes[1,0].fill_between(h_plot, 
                              props.min_temp-273.15, 
                              props.max_temp-273.15,
                              alpha=0.2, color='green', label='Safe')
        axes[1,0].axhline(y=props.min_temp-273.15, color='orange', 
                         linestyle='--', label='Min Limit')
        axes[1,0].axhline(y=props.max_temp-273.15, color='red', 
                         linestyle='--', label='Max Limit')
        axes[1,0].legend()
        axes[1,0].grid(True)

        cycles = np.linspace(0, 1000, 100)
        strength_degradation = props.base_strength * (props.fatigue_factor ** cycles)
        
        axes[1,1].plot(cycles, strength_degradation, 'brown')
        axes[1,1].set_xlabel('Thermal Cycles')
        axes[1,1].set_ylabel('Material Strength (MPa)')
        axes[1,1].set_title('Fatigue Life Estimation')
        axes[1,1].grid(True)
        axes[1,1].set_yscale('log')
        
        fig_stress.tight_layout()

        canvas_stress = FigureCanvasTkAgg(fig_stress, master=self.fig_frame_stress)
        canvas_stress.draw()
        canvas_stress.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def plot_all(self, h_vec, lift, inf, vb, V, m_e, m_f, m_bal, m_lg, res):
        for w in self.fig_frame_main.winfo_children(): w.destroy()
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))

        for i in range(3):
            for j in range(3):
                if i == 2 and j == 1:
                    axes[i, j] = fig.add_subplot(3, 3, i*3 + j + 1, projection='3d')
                else:
                    axes[i, j] = fig.add_subplot(3, 3, i*3 + j + 1)
        
        axes[0,0].plot(lift, h_vec, 'b'); axes[0,0].set_title("Net Lift (kN)")
        axes[0,1].plot(inf, h_vec, 'g'); axes[0,1].set_title("Inflation Fraction")
        axes[0,2].plot(vb, h_vec, 'm'); axes[0,2].set_title("Ballonet Volume (m³)")
        axes[1,0].plot(res['t'], h_vec, 'r'); axes[1,0].set_title("Ascent Profile (s)")
        axes[1,1].plot(res['flow'], h_vec, 'orange'); axes[1,1].set_title("Discharge Rate (m³/s)")
        
        v_gas = [V - v for v in vb]
        axes[1,2].plot(v_gas, h_vec, 'navy'); axes[1,2].set_title("Gas Volume (m³)")

        self.plot_shape_profile(axes[2, 1], V)
        self.plot_shape_profile_2d(axes[2, 0], V)
        axes[2, 2].axis('off')
        
        for ax in axes.flat: ax.grid(True); 
        if not isinstance(ax, Axes3D):
            ax.set_ylabel("alt (m)")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame_main)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.out_box.delete(1.0, tk.END)
        self.out_box.insert(tk.END, (
            f"--- SIZING RESULTS ---\n"
            f"Envelope Vol:  {V:.1f} m3\n"
            f"Envelope Mass: {m_e:.1f} kg\n"
            f"Fin Mass:      {m_f:.1f} kg\n"
            f"Ballonet Mass: {m_bal:.1f} kg\n"
            f"Ballonet Vol:  {max(vb):.1f} m3\n"
            f"Gas Mass:      {m_lg:.1f} kg\n"
            f"----------------------\n"
            f"Peak Discharge:{max(res['flow']):.4f} m3/s\n"
            f"Total Time:    {max(res['t']):.1f} s"
        ))
"""
    def compare_shapes(self):
    #Compare both shapes for the same requirements
    results = {}
    
    for shape in ["GNVR", "LYNX"]:
        self.geometry = AerostatGeometry(shape, self.p_fineness.get())
        self.Cd = self.geometry.get_drag_coefficient(10000)
        
        # Run sizing calculation
        V_env = self.solve_volume()  # Your existing volume solver
        
        L, D = self.geometry.get_dimensions(V_env)
        sa = self.geometry.get_surface_area(V_env)
        
        # Calculate key metrics
        drag_area = 0.25 * np.pi * D**2  # Frontal area
        drag_force = 0.5 * 1.225 * 10**2 * self.Cd * drag_area  # at 10 m/s wind
        
        results[shape] = {
            'Volume': V_env,
            'Length': L,
            'Diameter': D,
            'Surface Area': sa,
            'Drag Coefficient': self.Cd,
            'Drag Force (10m/s)': drag_force,
            'Fineness Ratio': L/D,
            'Shape Factor': sa / (4 * np.pi * (3*V_env/(4*np.pi))**(2/3))  # vs sphere
        }
    
    return results
"""


if __name__ == "__main__":
    root = tk.Tk(); app = FinalAerostatApp(root); root.mainloop()