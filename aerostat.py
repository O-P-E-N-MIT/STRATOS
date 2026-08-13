import numpy as np
from scipy.optimize import minimize_scalar, fsolve, least_squares

# The ISA atmospheric model used here is modelled only for Troposphere and Tropopause which is enough for our case.
T0 = 288.15             # Ground Temperature
P0 = 101325             # Ground Pressure
rho0 = 1.225            # Ground Density
ag = 9.80665             # Acceleration due to gravity at sea level
R  = 287                # Universal gas constant for air
RE = 6371e3             # Relative Humidity
K = rho0 * ag * T0/P0   # Aerostatic lift constant
RDWV = 0.622            # Relative density of water vapour
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant
CP = 1005.0             # Specific heat capacity at constant pressure for air.

# Temperature gradients (dT/dh in K/m)
L0 = -0.0065  # Troposphere cooling
L2 =  0.0010  # Mid Stratosphere warming
L3 =  0.0028  # Upper Stratosphere warming

# Base Pressures
P0, P1, P2, P3 = 101325.0, 22632.1, 5474.9, 868.0

# Base Temperatures
T0, T1, T2, T3 = 288.15, 216.65, 216.65, 228.65

# Base altitudes
h0, h1, h2, h3 = 0.0, 11000.0, 20000.0, 32000.0

# Factors to account for the surface area of the ballonet.
BALLONET_SHAPE_FACTOR = {
    "HEMISPHERE": 3 ** (2/3) * 2 ** (1/3),
    "THREE_QUARTER": 3
}

def get_atmospheric_properties(z):
    z = np.asarray(z)
    h = (RE * z) / (RE + z)  # Geopotential altitude

    T = np.empty_like(h, dtype=float)
    P = np.empty_like(h, dtype=float)

    tropo  = h < h1
    iso1   = (h >= h1) & (h < h2)  # Tropopause / Lower Stratosphere
    strat1 = (h >= h2) & (h < h3)  # Mid Stratosphere
    strat2 = h >= h3               # Upper Stratosphere (up to 47km)

    T[tropo] = T0 + L0 * h[tropo]
    P[tropo] = P0 * (T[tropo] / T0)**(-ag / (R * L0))

    T[iso1] = T1
    P[iso1] = P1 * np.exp(-ag * (h[iso1] - h1) / (R * T1))

    T[strat1] = T2 + L2 * (h[strat1] - h2)
    P[strat1] = P2 * (T[strat1] / T2)**(-ag / (R * L2))

    T[strat2] = T3 + L3 * (h[strat2] - h3)
    P[strat2] = P3 * (T[strat2] / T3)**(-ag / (R * L3))

    return P, T

def get_vapour_pressure (T, RH):
    Tc = T - 273.15
    e_sat = 611.21 * np.exp((18.678 - Tc/234.5)*(Tc/(257.14 + Tc)))
    return RH * e_sat

def get_net_lift (
        volume,
        total_mass,
        operational_height,
        RH,
        purity,
        delta_P,
        delta_T,
        gas_constant,
        inflation_fraction_factor
):
    P, T = get_atmospheric_properties(operational_height)
    e = get_vapour_pressure(T, RH)

    # Corrected density calculation using volumetric mixture rule
    rho_pure_lg = (P + delta_P) / (gas_constant * (T + delta_T))
    rho_contaminant = (P + delta_P) / (R * (T + delta_T))
    rho_lg = purity * rho_pure_lg + (1.0 - purity) * rho_contaminant

    rho_ba = P/(287*T)

    inflation_fraction = inflation_fraction_factor * ((T + delta_T) / (P + delta_P))

    Lg = K * volume * (P - (1-RDWV)*e) / T

    return Lg - (rho_lg * inflation_fraction * volume + rho_ba * (1 - inflation_fraction) * volume + total_mass) * ag

def get_gas_mass (
        P, T,
        volume,
        RH,
        purity,
        delta_P,
        delta_T,
        gas_constant,
        inflation_fraction_factor
):
    # Corrected density calculation using volumetric mixture rule
    rho_pure_lg = (P + delta_P) / (gas_constant * (T + delta_T))
    rho_contaminant = (P + delta_P) / (R * (T + delta_T))
    rho_lg = purity * rho_pure_lg + (1.0 - purity) * rho_contaminant

    rho_ba = P/(287*T)
    inflation_fraction = inflation_fraction_factor * ((T + delta_T) / (P + delta_P))

    return rho_lg * inflation_fraction * volume + rho_ba * (1 - inflation_fraction) * volume

def get_thermal_model (T_amb, solar_flux, absorptivity, emissivity, wind_speed):
    h_conv = 10.45 - wind_speed + 10.0 * np.sqrt(wind_speed)
    q_solar = solar_flux * absorptivity
    q_ir = emissivity * 5.67e-8 * (T_amb ** 4)
    T_env = T_amb + (q_solar - q_ir) / max(h_conv, 1e-3)

    return T_env

class AerostatHull:

    def __init__(
            self,
            envelope,
            skin_density,
            skin_thickness,
            additional_mass,
            operational_height,
            deployment_height,
            margin_height,
            RH,
            purity,
            delta_P,
            delta_T,
            gas_constant,
            gamma=5/3,
            inflation_fraction_oper=0.9,
            albedo=0.2,
            earth_temperature=T0,
            lobe_number=1,
            e=0, f=0, g=0,
            fin_rc=0,
            fin_taper_ratio=1,
            fin_height=0,
            fin_thickness=0,
            fin_density=0,
            fin_number=1,
            has_wings=False,
            wing_span=0,
            wing_root_chord=0,
            wing_tip_chord=0,
            wing_thickness=0,
            wing_density=10.0,
            ballonet_number=2,
            ballonet_shape="THREE_QUARTER",
            ballonet_fabric_density=0.35,
            tether_density=0,
            tether_fraction=1,
            cte=2.3e-5,
            max_temp=323.15,
            min_temp=233.15,
            base_strength=75.0,
            temp_derating=0.15,
            fatigue_factor=0.995,
            uv_degradation=0.02,
            solar_flux=1000,
            emissivity=0.8,
            absorptivity=0.3,
            wind_speed=5
    ):
        P_dep, T_dep = get_atmospheric_properties(deployment_height)
        P_op,  T_op  = get_atmospheric_properties(operational_height)

        if ballonet_number == 0:
            self.inflation_fraction_oper = 1
            self.inflation_fraction_deploy = 1
            self.inflation_fraction_factor = UnitMultiplier()
            self.has_ballonets = False

        else:
            self.inflation_fraction_oper = inflation_fraction_oper
            self.inflation_fraction_deploy = inflation_fraction_oper * ((P_op + delta_P) / (P_dep + delta_P)) * ((T_dep + delta_T) / (T_op + delta_T))
            self.inflation_fraction_factor = inflation_fraction_oper * (P_op + delta_P) / (T_op + delta_T)
            self.has_ballonets = True

        self.delta_P = delta_P
        self.delta_T = delta_T
        self.deployment_altitude = deployment_height
        self.operational_altitude = operational_height
        self.pressure_altitude = margin_height + operational_height
        self.envelope = envelope
        self.gas_properties = (RH, purity, delta_P, delta_T, gas_constant, self.inflation_fraction_factor)
        self.gamma = gamma
        self.lobe_number = lobe_number
        self.multi_lobe_distances = (e, f, g)
        self.skin_density = skin_density
        self.additional_mass = additional_mass
        self.has_ballonets = ballonet_number != 0

        self.cte = cte
        self.base_strength = base_strength
        self.temp_derating = temp_derating
        self.skin_thickness = skin_thickness
        self.solar_flux = solar_flux
        self.earth_temperature = earth_temperature
        self.albedo = albedo
        self.emissivity = emissivity
        self.absorptivity = absorptivity
        self.wind_speed = wind_speed

        self.tether_density = tether_density * tether_fraction
        self.ballonet_fabric_mass = BALLONET_SHAPE_FACTOR.get(ballonet_shape, 3) * ballonet_fabric_density * (np.pi * ballonet_number)**(1/3) * (1 - self.inflation_fraction_deploy)**(2/3)
        self.fin_mass = 0.0393 * fin_thickness*1e-2 * fin_rc**2 * fin_height * fin_density * (fin_taper_ratio + (fin_taper_ratio - 1)**2 / 3) * fin_number

        self.wing_mass = 0
        if has_wings:
            total_planform = 0.5 * (wing_root_chord + wing_tip_chord) * wing_span
            avg_chord = (wing_root_chord + wing_tip_chord) / 2
            wing_vol = total_planform * (avg_chord * (wing_thickness / 100.0))
            self.wing_mass = wing_vol * wing_density

    def get_properties (self, n=None, include_tether=True):
        if n is None:
            n = int((self.pressure_altitude - self.deployment_altitude) / 100)

        h = np.linspace(self.deployment_altitude, self.pressure_altitude, n)

        e, f, g = self.multi_lobe_distances
        RH, purity, delta_P, delta_T, gas_constant, _ = self.gas_properties

        P, T = get_atmospheric_properties(h)
        e_vap = get_vapour_pressure(T, RH)
        rho = P / (R * T)

        if self.lobe_number == 1:
            volume = self.envelope.volume()
            surface_area = self.envelope.surface_area()
            projected_area = self.envelope.side_projected_area()
        elif self.lobe_number == 2:
            volume = self.envelope.volume_bilobe(f)
            surface_area = self.envelope.surface_area_bilobe(f)
            projected_area = self.envelope.top_projected_area_bilobe(f)
        else:
            volume = self.envelope.volume_trilobe(e, f, g)
            surface_area = self.envelope.surface_area_trilobe(e, f, g)
            projected_area = self.envelope.top_projected_area_trilobe(e, f, g)

        if self.has_ballonets:
            I = self.inflation_fraction_factor * ((T + delta_T) / (P + delta_P))
            I = np.clip(I, 0, 1)
        else:
            I = np.full_like(P, 1)

        current_tether_mass = (self.tether_density * h) if include_tether else 0

        ballonet_mass = self.ballonet_fabric_mass * volume**(2/3)

        total_mass = (self.skin_density * surface_area +
                      self.additional_mass +
                      self.fin_mass +
                      self.wing_mass +
                      current_tether_mass +
                      ballonet_mass)

        BV = (1 - I) * volume

        # Corrected density calculation using volumetric mixture rule
        rho_pure_lg = (P + delta_P) / (gas_constant * (T + delta_T))
        rho_contaminant = (P + delta_P) / (R * (T + delta_T))
        rho_lg = purity * rho_pure_lg + (1.0 - purity) * rho_contaminant

        rho_ba = rho

        Lg = K * volume * (P - (1-RDWV)*e_vap) / T

        Ln = Lg - (rho_lg * I * volume + rho_ba * (1 - I) * volume + total_mass) * ag

        mu = 1.458e-6 * (T**1.5) / (T + 110.4)
        k = 0.024 * (T / 293.15)**0.8

        cp_lg = (self.gamma / (self.gamma - 1.0)) * gas_constant
        mu_lg = 1.9e-5 * (T / 293.15)**0.7
        k_lg = 0.15 * (T / 293.15)**0.7

        # Corrected specific heat and gas constant using mass fractions
        mass_fraction_lg = (purity * rho_pure_lg) / rho_lg
        mass_fraction_air = ((1.0 - purity) * rho_contaminant) / rho_lg

        cp_mix = mass_fraction_lg * cp_lg + mass_fraction_air * CP
        R_mix = mass_fraction_lg * gas_constant + mass_fraction_air * R

        mu_mix = purity * mu_lg + (1.0 - purity) * mu
        k_mix = purity * k_lg + (1.0 - purity) * k

        nu = mu / rho
        Pr = (CP * mu) / k
        D = self.envelope.diameter

        # T_env is evaluated first to prevent UnboundLocalError
        T_env = get_thermal_model(T, self.solar_flux, self.absorptivity, self.emissivity, self.wind_speed)

        # Clamp Re and h_o to prevent negative convective coefficients which flip the Jacobian
        if self.wind_speed > 0:
            Re = (rho * self.wind_speed) / mu
            h_o = (k / D) * Re * Pr * (0.2275 / (np.log10(np.maximum(Re, 10))**2.584) - 850.0 / np.maximum(Re, 10))
            h_o = np.maximum(h_o, 1e-3)
        else:
            Gr_a = (ag * (1/T) * abs(T - T_env) * D**3) / (nu**2)
            h_o = (k / D) * (0.6 + 0.387 * ((Gr_a * Pr) / (1.0 + (0.559 / Pr)**(9/16))**(16/9))**(1/6))**2
            h_o = np.maximum(h_o, 1e-3)

        T_bb = 0.052 * T ** 1.5

        A = surface_area / 2
        m = A * self.skin_density

        Qd_u = self.absorptivity * self.solar_flux * projected_area
        Qd_l = self.absorptivity * self.solar_flux * self.albedo * projected_area
        Qsky_u_const = self.emissivity * A * SIGMA * T_bb ** 4
        Qsky_l_const = self.emissivity * A * SIGMA * T_bb ** 4
        Qearth_l_const = self.emissivity * A * SIGMA * self.earth_temperature ** 4

        T_g = np.zeros(n)
        T_u = np.zeros(n)
        T_l = np.zeros(n)

        T_LOW, T_HIGH = 100.0, 400.0

        # Offset the initial guess to prevent derivative explosion at T_g = T_u = T_l
        current_guess = np.clip(np.array([T[0] + 1.0, T[0] + 2.0, T[0] - 1.0]), T_LOW, T_HIGH)

        for i in range(n):
            Qd_u_i = Qd_u[i] if isinstance(Qd_u, np.ndarray) else Qd_u
            Qd_l_i = Qd_l[i] if isinstance(Qd_l, np.ndarray) else Qd_l
            Qsky_u_const_i = Qsky_u_const[i] if isinstance(Qsky_u_const, np.ndarray) else Qsky_u_const
            Qsky_l_const_i = Qsky_l_const[i] if isinstance(Qsky_l_const, np.ndarray) else Qsky_l_const
            Qearth_l_const_i = Qearth_l_const[i] if isinstance(Qearth_l_const, np.ndarray) else Qearth_l_const

            residual_scale = max(h_o[i] * A, 1.0)

            def steady_thermal_single(T_vars):
                T_g_var, T_u_var, T_l_var = T_vars

                # Corrected to account for structural superpressure
                rho_g = (P[i] + self.delta_P) / (R_mix[i] * T_g_var)
                nu_g = mu_mix[i] / rho_g
                beta_g = 1.0 / T_g_var
                Pr_g = (cp_mix[i] * mu_mix[i]) / k_mix[i]

                # Added +1e-3 safely inside the np.abs() to ensure finite Jacobians at T_diff = 0
                Gr_g_u = (ag * beta_g * (np.abs(T_g_var - T_u_var) + 1e-3) * D**3) / (nu_g**2 + 1e-12)
                Gr_g_l = (ag * beta_g * (np.abs(T_g_var - T_l_var) + 1e-3) * D**3) / (nu_g**2 + 1e-12)
                h_i_u = 0.13 * (k_mix[i] / D) * np.abs(Gr_g_u * Pr_g)**0.33
                h_i_l = 0.13 * (k_mix[i] / D) * np.abs(Gr_g_l * Pr_g)**0.33

                IR_l_to_u = self.emissivity * A * SIGMA * (T_l_var**4 - T_u_var**4)
                IR_u_to_l = self.emissivity * A * SIGMA * (T_u_var**4 - T_l_var**4)

                Q_sky_u = Qsky_u_const_i - self.emissivity * A * SIGMA * T_u_var**4
                Q_sky_l = Qsky_l_const_i - self.emissivity * A * SIGMA * T_l_var**4

                Q_cv_a_u = h_o[i] * A * (T[i] - T_u_var)
                Q_cv_a_l = h_o[i] * A * (T[i] - T_l_var)

                Q_cv_g_u = h_i_u * A * (T_g_var - T_u_var)
                Q_cv_g_l = h_i_l * A * (T_g_var - T_l_var)

                dTu = Q_cv_a_u + Q_cv_g_u + Q_sky_u + IR_l_to_u + Qd_u_i
                dTl = Q_cv_a_l + Q_cv_g_l + Q_sky_l + IR_u_to_l + Qd_l_i + Qearth_l_const_i
                dTg = -Q_cv_g_u - Q_cv_g_l

                return np.array([dTg, dTu, dTl]) / residual_scale

            result = least_squares(
                steady_thermal_single,
                current_guess,
                bounds=(T_LOW, T_HIGH),
                method='trf',
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=2000,
            )

            sol = result.x
            converged = result.success and np.max(np.abs(result.fun)) < 1e-6

            if not converged:
                retry_guess = np.clip(np.array([T[i] + 1.0, T[i] + 2.0, T[i] - 1.0]), T_LOW, T_HIGH)
                retry = least_squares(
                    steady_thermal_single,
                    retry_guess,
                    bounds=(T_LOW, T_HIGH),
                    method='trf',
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    max_nfev=5000,
                )

                if np.max(np.abs(retry.fun)) < np.max(np.abs(result.fun)):
                    result = retry
                    sol = result.x

                T_g[i], T_u[i], T_l[i] = sol
                current_guess = sol

            else:
                T_g[i], T_u[i], T_l[i] = sol
                current_guess = sol

        # Corrected to use Hoop Stress (2 * t) instead of spherical stress (4 * t)
        sigma = (
                self.cte * (T_env - T) * self.base_strength
                + delta_P * self.envelope.diameter / (2 * self.skin_thickness) * 1e-6
        )

        derating = np.full_like(T_env, 1)
        derating_mask = T_env > 293.15
        derating[derating_mask] = np.maximum(0, 1 - (T_env[derating_mask] - 293.15) * self.temp_derating / 100)
        sigma *= derating

        return h, Ln, Lg, I, BV, T_g, T_u, T_l, sigma, volume, surface_area

    def initialise_from_operational_altitude(self, bounds, target_lift=0.0):
        P_op, T_op = get_atmospheric_properties(self.operational_altitude)
        RH, purity, delta_P, delta_T, gas_constant, _ = self.gas_properties
        e_vap = get_vapour_pressure(T_op, RH)

        I_op = self.inflation_fraction_oper

        rho_pure_lg = (P_op + delta_P) / (gas_constant * (T_op + delta_T))
        rho_contaminant = (P_op + delta_P) / (R * (T_op + delta_T))
        rho_lg = purity * rho_pure_lg + (1.0 - purity) * rho_contaminant

        rho_ba = P_op / (287 * T_op)

        tether_mass_op = self.tether_density * self.operational_altitude

        def objective(L):
            self.envelope.set_length(L)

            if self.lobe_number == 1:
                vol = self.envelope.volume()
                surf = self.envelope.surface_area()
            elif self.lobe_number == 2:
                vol = self.envelope.volume_bilobe(self.multi_lobe_distances[1])
                surf = self.envelope.surface_area_bilobe(self.multi_lobe_distances[1])
            else:
                vol = self.envelope.volume_trilobe(*self.multi_lobe_distances)
                surf = self.envelope.surface_area_trilobe(*self.multi_lobe_distances)

            ballonet_mass = self.ballonet_fabric_mass * vol**(2/3)

            wing_m = getattr(self, 'wing_mass', 0)

            total_mass = (self.skin_density * surf +
                          self.additional_mass +
                          self.fin_mass +
                          wing_m +
                          tether_mass_op +
                          ballonet_mass)

            Lg = K * vol * (P_op - (1-RDWV)*e_vap) / T_op
            Ln = Lg - (rho_lg * I_op * vol + rho_ba * (1 - I_op) * vol + total_mass) * ag

            return abs(Ln - target_lift)

        search_bounds = (max(1.0, bounds[0]), min(1000.0, bounds[1]))
        res = minimize_scalar(objective, bounds=search_bounds, method='bounded', options={'xatol': 1e-4})

        self.envelope.set_length(res.x)
        return self.envelope, res.fun

    def get_burst_altitude (self, safety_factor=2):
        allowable_stress = self.base_strength / safety_factor

        # Corrected to use Hoop Stress (2 * t) instead of spherical stress (4 * t)
        hoop_stress_factor = self.envelope.diameter / (2 * self.skin_thickness)

        def func (h):
            _, T = get_atmospheric_properties(h)
            T_env = get_thermal_model(T, self.solar_flux, self.absorptivity, self.emissivity, self.wind_speed)

            thermal_strain = self.cte * (T_env - T)
            thermal_stress = thermal_strain * self.base_strength

            sigma_pa = self.delta_P * hoop_stress_factor
            pressure_stress = sigma_pa * 1e-6

            total_stress = thermal_stress + pressure_stress

            if T_env > 293.15:
                total_stress *= max(0, 1 - (T_env - 293.15) * self.temp_derating / 100)

            return allowable_stress - total_stress

        try:
            h_burst = minimize_scalar(func, bounds=[0, 20000], method='bounded', options={'xatol': 1e-8})
            return h_burst.x
        except ValueError:
            return 20000

class UnitMultiplier:
    def __mul__(self, other):
        return 1

    def __rmul__(self, other):
        return 1
