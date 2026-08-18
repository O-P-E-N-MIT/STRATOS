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

# TODO: Right now, there is only support for ISA model only till the Tropopause.
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
        volume,                     # Volume of the envelope.
        total_mass,                 # Fixed mass of the aerostat.
        operational_height,         # Operational altitude of the envelope.
        RH,                         # Relative Humidity (0-1).
        purity,                     # Purity of lifting gas.
        delta_P,                    # Increment in lifting gas pressure.
        delta_T,                    # Increment in lifting gas temperature.
        gas_constant,               # Gas constant for the gas filled in the aerostat.
        inflation_fraction_factor   # Inflation Fraction factor.
):
    P, T = get_atmospheric_properties(operational_height)
    e = get_vapour_pressure(T, RH)

    # The formulae used in the MATLAB (which is also implemented here) is different from that of resources given by sir.
    # TODO: Correct these formulae.
    rho_lg = purity * (P + delta_P) / (gas_constant * (T + delta_T))
    rho_ba = P/(287*T)

    # Get the inflation fraction at that altitude.
    inflation_fraction = inflation_fraction_factor * ((T + delta_T) / (P + delta_P))

    # Gross static lift
    Lg = K * volume * (P - (1-RDWV)*e) / T

    # Net static lift
    return Lg - (rho_lg * inflation_fraction * volume + rho_ba * (1 - inflation_fraction) * volume + total_mass) * ag

def get_gas_mass (
        P, T,
        volume,                     # Volume of the envelope.
        RH,                         # Relative Humidity (0-1).
        purity,                     # Purity of lifting gas.
        delta_P,                    # Increment in lifting gas pressure.
        delta_T,                    # Increment in lifting gas temperature.
        gas_constant,               # Gas constant for the gas filled in the aerostat.
        inflation_fraction_factor   # Inflation Fraction factor.
):
    rho_lg = purity * (P + delta_P) / (gas_constant * (T + delta_T))
    rho_ba = P/(287*T)
    inflation_fraction = inflation_fraction_factor * ((T + delta_T) / (P + delta_P))

    # Returns the lifting gas mass and the ballonet mass
    return rho_lg * inflation_fraction * volume + rho_ba * (1 - inflation_fraction) * volume


# NOTE: This is the older thermal model used by STRATOS to get envelope temperature.
def get_thermal_model (T_amb, solar_flux, absorptivity, emissivity, wind_speed):
    # This formula is only valid for small characteristic lengths.
    h_conv = 10.45 - wind_speed + 10.0 * np.sqrt(wind_speed)
    q_solar = solar_flux * absorptivity
    q_ir = emissivity * 5.67e-8 * (T_amb ** 4)
    T_env = T_amb + (q_solar - q_ir) / max(h_conv, 1e-3)

    return T_env

# Main class to perform all calculations for the Aerostat.
class AerostatHull:

    def __init__(
            self,
            envelope,                       # The envelope to be modelled as Aerostat.
            skin_density,                   # Density of the skin of the hull (kg/m^2).
            skin_thickness,                 # Thickness of the envelope skin.
            additional_mass,                # Additional mass of the envelope.
            operational_height,             # Operational altitude of the envelope.
            deployment_height,              # Deployment altitude of the envelope.
            margin_height,                  # Margin for the pressure altitude.
            RH,                             # Relative Humidity (0-1).
            purity,                         # Purity of lifting gas.
            delta_P,                        # Increment in lifting gas pressure.
            delta_T,                        # Increment in lifting gas temperature.
            gas_constant,                   # Gas constant for the gas filled in the aerostat.
            gamma=5/3,                      # Adiabatic index of the lifting gas.
            inflation_fraction_oper=0.9,    # Inflation Fraction at operation.
            albedo=0.2,                     # Reflectivity of Earth's surface.
            earth_temperature=T0,           # Temperature of Earth's surface.
            lobe_number=1,                  # Lobe number
            e=0, f=0, g=0,                  # Lobe offsets
            fin_rc=0,                       # Root chord of the fin.
            fin_taper_ratio=1,              # Taper ratio of the fin.
            fin_height=0,                   # Height of the fin.
            fin_thickness=0,                # Ratio of fin thickness to chord ratio of the NACA airfoil to be used.
            fin_density=0,                  # Density of the fin material (kg/m^3).
            fin_number=1,                   # Fin number.
            has_wings=False,                # Wing toggle.
            wing_span=0,                    # Wing span.
            wing_root_chord=0,              # Wing root chord.
            wing_tip_chord=0,               # Wing tip chord.
            wing_thickness=0,               # Wing thickness %.
            wing_density=10.0,              # Wing density (kg/m^3).
            ballonet_number=2,              # Number of ballonets.
            ballonet_shape="THREE_QUARTER", # Ballonet shape.
            ballonet_fabric_density=0.35,   # Ballonet fabric density (kg/m^2).
            tether_density=0,               # Density of the tether used (kg/m).
            tether_fraction=1,              # Fraction of tether weight carried.
            cte=2.3e-5,                     # Coefficient of thermal expansion.
            max_temp=323.15,
            min_temp=233.15,
            elastic_modulus=75.0,           # Elastic modulus of envelope (MPa)
            strength=75.0,                  # Maximum (yield) strength of envelope (MPa)
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

        # In case if there are no ballonets, the inflation fraction is always 1.
        if ballonet_number == 0:
            self.inflation_fraction_oper = 1
            self.inflation_fraction_deploy = 1
            self.inflation_fraction_factor = UnitMultiplier()
            self.has_ballonets = False

        # If there are ballonets, the necessary inflation fraction calculations are to be done.
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
        self.elastic_modulus = elastic_modulus
        self.strength = strength
        self.temp_derating = temp_derating
        self.skin_thickness = skin_thickness
        self.solar_flux = solar_flux
        self.earth_temperature = earth_temperature
        self.albedo = albedo
        self.emissivity = emissivity
        self.absorptivity = absorptivity
        self.wind_speed = wind_speed

        # Tether weight per unit meter.
        self.tether_density = tether_density * tether_fraction

        # Ballonet fabric mass per unit volume of envelope^2/3
        self.ballonet_fabric_mass = BALLONET_SHAPE_FACTOR.get(ballonet_shape, 3) * ballonet_fabric_density * (np.pi * ballonet_number)**(1/3) * (1 - self.inflation_fraction_deploy)**(2/3)

        # Fin mass calculation
        self.fin_mass = 0.0393 * fin_thickness*1e-2 * fin_rc**2 * fin_height * fin_density * (fin_taper_ratio + (fin_taper_ratio - 1)**2 / 3) * fin_number

        # Wing mass calculation
        self.wing_mass = 0
        if has_wings:
            total_planform = 0.5 * (wing_root_chord + wing_tip_chord) * wing_span
            avg_chord = (wing_root_chord + wing_tip_chord) / 2
            wing_vol = total_planform * (avg_chord * (wing_thickness / 100.0))
            self.wing_mass = wing_vol * wing_density

    def get_properties (self, n=None, include_tether=True, safety_factor=4.0):
        if n is None:
            n = int((self.pressure_altitude - self.deployment_altitude) / 100)

        D = self.envelope.diameter
        e, f, g = self.multi_lobe_distances
        
        h = np.linspace(self.deployment_altitude, self.pressure_altitude, n)
        RH, purity, delta_P, delta_T, gas_constant, _ = self.gas_properties

        P, T = get_atmospheric_properties(h)
        e_vap = get_vapour_pressure(T, RH)
        rho = P / (R * T)
        mu = 1.458e-6 * (T**1.5) / (T + 110.4)  
        k = 0.024 * (T / 293.15)**0.8    
        nu = mu / rho
        Pr = (CP * mu) / k    

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

        current_tether_mass = (self.tether_density * h) if include_tether else 0
        ballonet_mass = self.ballonet_fabric_mass * volume**(2/3)
        total_mass = (self.skin_density * surface_area +
                      self.additional_mass +
                      self.fin_mass +
                      self.wing_mass +
                      current_tether_mass +                
                      ballonet_mass) 

        # NOTE: These formulae are only valid for Helium lifting gas.
        cp_lg = (self.gamma / (self.gamma - 1.0)) * gas_constant
        mu_lg = 1.9e-5 * (T / 293.15)**0.7 
        k_lg = 0.15 * (T / 293.15)**0.7

        # Due to presence of ballonet air, the properties of the net mixture is computed by linear blending.
        cp_mix = purity * cp_lg + (1 - purity) * CP
        mu_mix = purity * mu_lg + (1 - purity) * mu
        k_mix = purity * k_lg + (1 - purity) * k
        R_mix = purity * gas_constant + (1 - purity) * R
        
        # In case of forced convection,
        if self.wind_speed > 0:
            Re = (rho * self.wind_speed) / mu
            h_o_forced = (k / D) * Re * Pr * (0.2275 / (np.log10(Re)**2.584) - 850.0 / Re) 

        # Black body temperature
        T_bb = 0.052 * T ** 1.5

        # Assuming the envelope is symmetric about the yaw plane, the surface area of the upper half and the lower half
        # of envelope must be 0.
        #
        # NOTE: This is not true for asymmetric envelopes and asymmetric trilobe configurations.
        thermal_element_area = surface_area / 2
        
        Qd_u = self.absorptivity * self.solar_flux * projected_area
        Qd_l = self.absorptivity * self.solar_flux * self.albedo * projected_area
        Qsky_u0 = self.emissivity * thermal_element_area * SIGMA * T_bb ** 4
        Qsky_l0 = self.emissivity * thermal_element_area * SIGMA * T_bb ** 4
        Qearth_l = self.emissivity * thermal_element_area * SIGMA * self.earth_temperature ** 4

        T_g = np.zeros(n)
        T_u = np.zeros(n)
        T_l = np.zeros(n)

        T_guess = np.array([T[0], T[0], T[0]])

        for i in range(n):

            def steady_thermal_single (T_vars):
                T_g_var, T_u_var, T_l_var = T_vars

                rho_g = P[i] / (R_mix * T_g_var)
                nu_g = mu_mix[i] / rho_g
                beta_g = 1.0 / T_g_var
                Pr_g = (cp_mix * mu_mix[i]) / k_mix[i]

                # In case of natural convection,
                if self.wind_speed == 0:
                    Gr_a_u = (ag * (1/T) * np.abs(T_g_var - T_u_var) * D**3) / (nu**2)
                    Gr_a_l = (ag * (1/T) * np.abs(T_g_var - T_l_var) * D**3) / (nu**2)
                    h_o_l = (k / D) * (0.6 + 0.387 * ((Gr_a_l * Pr) / (1.0 + (0.559 / Pr)**(9/16))**(16/9))**(1/6))**2 
                    h_o_u = (k / D) * (0.6 + 0.387 * ((Gr_a_u * Pr) / (1.0 + (0.559 / Pr)**(9/16))**(16/9))**(1/6))**2 

                # In case of forced convection,
                else:
                    h_o_l = h_o_u = h_o_forced

                Gr_g_u = (ag * beta_g * np.abs(T_g_var - T_u_var) * D**3) / (nu_g**2 + 1e-12)
                Gr_g_l = (ag * beta_g * np.abs(T_g_var - T_l_var) * D**3) / (nu_g**2 + 1e-12)
                h_i_u = 0.13 * (k_mix[i] / D) * np.abs(Gr_g_u * Pr_g)**0.33
                h_i_l = 0.13 * (k_mix[i] / D) * np.abs(Gr_g_l * Pr_g)**0.33

                # Infrared emissivity
                IR_l_to_u = self.emissivity * thermal_element_area * SIGMA * (T_l_var**4 - T_u_var**4)
                IR_u_to_l = self.emissivity * thermal_element_area * SIGMA * (T_u_var**4 - T_l_var**4)

                # Black body radiation.
                Q_sky_u = Qsky_u0[i] - self.emissivity * thermal_element_area * SIGMA * T_u_var**4
                Q_sky_l = Qsky_l0[i] - self.emissivity * thermal_element_area * SIGMA * T_l_var**4

                # Natural convection from atmosphere.
                Q_cv_a_u = h_o_u[i] * thermal_element_area * (T[i] - T_u_var)
                Q_cv_a_l = h_o_l[i] * thermal_element_area * (T[i] - T_l_var)

                # Internal convection from the lifting gas.
                Q_cv_g_u = h_i_u * thermal_element_area * (T_g_var - T_u_var)
                Q_cv_g_l = h_i_l * thermal_element_area * (T_g_var - T_l_var)

                dQ_u = Q_cv_a_u + Q_cv_g_u + Q_sky_u + IR_l_to_u + Qd_u
                dQ_l = Q_cv_a_l + Q_cv_g_l + Q_sky_l + IR_u_to_l + Qd_l + Qearth_l
                dQ_g = -Q_cv_g_u - Q_cv_g_l

                # NOTE: A residual scaler has to be added so that in case of smaller airships with smaller areas 
                # don't end up converging properly. Something like a factor of 1/max(h0 * A, 1).
                return np.array([dQ_g, dQ_u, dQ_l])

            result = least_squares(
                steady_thermal_single,
                T_guess,
                bounds=(100, 400),
                method='trf',
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=2000,
            )

            sol = result.x
            converged = result.success and np.max(np.abs(result.fun)) < 1e-6

            T_g[i], T_u[i], T_l[i] = sol
            T_guess = sol

            if not converged:
                break

        T_star = T_g if delta_T == None else (T + delta_T)

        if self.has_ballonets:
            I = self.inflation_fraction_factor * (T_star / (P + delta_P))
            I = np.clip(I, 0, 1)
        else:
            I = np.full_like(P, 1)
        
        BV = (1 - I) * volume

        rho_lg = purity * (P + delta_P) / (gas_constant * T_star)
        rho_ba = rho
        
        Lg = K * volume * (P - (1-RDWV)*e_vap) / T
        Ln = Lg - (rho_lg * I * volume + rho_ba * (1 - I) * volume + total_mass) * ag

        # T_env = get_thermal_model(T, self.solar_flux, self.absorptivity, self.emissivity, self.wind_speed)
        
        # sigma = (
        #     self.cte * (T_env - T) * self.elastic_modulus                                 
        #     + delta_P * self.envelope.diameter / (4 * self.skin_thickness) * 1e-6   
        # )

        sigma = (
            self.cte * np.abs(T_u - T_l) * self.elastic_modulus / 2                                
            + delta_P * self.envelope.diameter / (4 * self.skin_thickness) * 1e-6   
        )

        T_env = (T_u + T_l) / 2
        
        derating = np.full_like(T_env, 1)
        derating_mask = T_env > 293.15
        derating[derating_mask] = np.maximum(0, 1 - (T_env[derating_mask] - 293.15) * self.temp_derating / 100)
        sigma *= derating

        idx = np.searchsorted(sigma, self.strength / safety_factor)

        if idx == 0:
            burst_altitude = h[0]
        elif idx == n-1:
            burst_altitude = h[n-1]
        else:
            burst_altitude = h[idx-1]

        return h, Ln, Lg, I, BV, T_g, T_u, T_l, sigma, volume, surface_area, burst_altitude, converged

    def initialise_from_operational_altitude(self, bounds, target_lift=0.0):
        # Extract static atmospheric properties at operational altitude
        P_op, T_op = get_atmospheric_properties(self.operational_altitude)
        RH, purity, delta_P, delta_T, gas_constant, _ = self.gas_properties
        e_vap = get_vapour_pressure(T_op, RH)

        I_op = self.inflation_fraction_oper
        rho_lg = purity * (P_op + delta_P) / (gas_constant * (T_op + delta_T))
        rho_ba = P_op / (287 * T_op)

        tether_mass_op = self.tether_density * self.operational_altitude

        def objective(L):
            self.envelope.set_length(L)

            # Calculate volume and surface area dynamically as length changes
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

            # Support wings safely
            wing_m = getattr(self, 'wing_mass', 0)

            total_mass = (self.skin_density * surf +
                          self.additional_mass +
                          self.fin_mass +
                          wing_m +
                          tether_mass_op +
                          ballonet_mass)

            Lg = K * vol * (P_op - (1-RDWV)*e_vap) / T_op
            Ln = Lg - (rho_lg * I_op * vol + rho_ba * (1 - I_op) * vol + total_mass) * ag

            # We want to minimize the absolute difference between actual lift and target lift
            return abs(Ln - target_lift)

        # Cap the max search length to 1000m to prevent OverflowErrors during optimization
        search_bounds = (max(1.0, bounds[0]), min(1000.0, bounds[1]))

        # Minimize the difference to find the perfect length
        res = minimize_scalar(objective, bounds=search_bounds, method='bounded', options={'xatol': 1e-4})

        # Set the envelope to the newly optimized length and return
        self.envelope.set_length(res.x)
        return self.envelope, res.fun

    # This function calculates the burst altitude beyond the maximum operational altitude to find the factor of safety.
    # NOTE: Given the atmosphere is limited upto 20km, the burst altitude cannot be calculate beyond that.
    def get_burst_altitude (self, safety_factor=2):
        # To fix this.
        allowable_stress = self.strength / safety_factor
        hoop_stress_factor = self.envelope.diameter / (4 * self.skin_thickness)

        def func (h):
            _, T = get_atmospheric_properties(h)
            T_env = get_thermal_model(T, self.solar_flux, self.absorptivity, self.emissivity, self.wind_speed)

            # Thermal stress
            thermal_strain = self.cte * (T_env - T)
            thermal_stress = thermal_strain * self.elastic_modulus

            # Hoop stress from pressure difference (thin shell, approximate spherical/prolate).
            sigma_pa = self.delta_P * hoop_stress_factor
            pressure_stress = sigma_pa * 1e-6

            # Total stress acting on the envelope skin due to both thermal and pressure effects.
            total_stress = thermal_stress + pressure_stress

            # Temperature derating on material strength.
            if T_env > 293.15:
                total_stress *= max(0, 1 - (T_env - 293.15) * self.temp_derating / 100)

            return allowable_stress - total_stress

        try:
            h_burst = minimize_scalar(func, bounds=[0, 20000], method='bounded', options={'xatol': 1e-8})
            return h_burst.x
        # If the burst altitude is beyond 20km, it returns 20km as the burst altitude.
        except ValueError:
            return 20000

# A unique number which when multiplied by anything will end up giving 1.
class UnitMultiplier:

    def __mul__(self, other):
        return 1

    def __rmul__(self, other):
        return 1