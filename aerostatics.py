import numpy as np

from scipy.optimize import minimize_scalar

# The ISA atmospheric model used here is modelled only for Troposphere and Tropopause which is enough for our case.

T0 = 288.15             # Ground Temperature
P0 = 101325             # Ground Pressure
rho0 = 1.225            # Ground Density
g = 9.80665             # Acceleration due to gravity at sea level
L  = 0.0065             # Lapse rate in Troposphere
R  = 287                # Universal gas constant for air
RE = 6371e3             # Relative Humidity
K = rho0 * g * T0/P0    # Aerostatic lift constant
RDWV = 0.622            # Relative density of water vapour

# def get_atmospheric_properties(z):
#     # Geopotential height from geometric height
#     h = (R * z) / (R + z)

#     # Gradient layer
#     if h < 11000:
#         T = T0 - L*h
#         P = P0 * (T/T0)**(9.80665/(R*L))

#     # Isothermal layer
#     else:
#         T = 216.65
#         P = 22632 * math.exp(-9.80665*(h-11000)/(R*T))

#     return P, T

def get_atmospheric_properties(z):
    z = np.asarray(z)
    h = (R * z) / (R + z)

    # initialise outputs
    T = np.empty_like(h, dtype=float)
    P = np.empty_like(h, dtype=float)

    # masks
    grad = h < 11000
    iso  = ~grad

    # gradient region
    T[grad] = T0 - L*h[grad]
    P[grad] = P0 * (T[grad]/T0)**(g/(R*L))

    # isothermal region
    T[iso] = 216.65
    P[iso] = 22632 * np.exp(-g*(h[iso]-11000)/(R*216.65))

    return P, T

def get_vapour_pressure (T, RH):
    Tc = T - 273.15
    e_sat = 611.21 * np.exp((18.678 - Tc/234.5)*(Tc/(257.14 + Tc)))
    return RH * e_sat

def get_net_lift (
        volume,                 # Volume of the envelope.
        total_mass,             # Fixed mass of the aerostat.
        operational_height,     # Operational altitude of the envelope.
        RH,                     # Relative Humidity (0-1).   
        purity,                 # Purity of lifting gas.
        delta_P,                # Increment in lifting gas pressure.
        delta_T,                # Increment in lifting gas temperature.
        gas_constant,           # Gas constant for the gas filled in the aerostat.
        inflation_fraction      # Inflation Fraction.
):
    P, T = get_atmospheric_properties(operational_height)
    e = get_vapour_pressure(T, RH)

    # The formulae used in the MATLAB (which is also implemented here) is different from that of resources given by sir.
    # TODO: Correct these formulae.
    rho_lg = purity * (P + delta_P) / (gas_constant * (T + delta_T))
    rho_ba = P/(287*T)

    # Gross static lift
    Lg = K * volume * (P - (1-RDWV)*e) / T

    # Net static lift
    return Lg - (rho_lg * inflation_fraction * volume + rho_ba * (1 - inflation_fraction) * volume + total_mass) * g

class AerostatHull:

    def __init__(
            self,
            envelope,                       # The envelope to be modelled as Aerostat.
            additional_mass,                # Additional mass of the envelope.
            skin_density,                   # Density of the skin of the hull (kg/m^3).
            operational_height,             # Operational altitude of the envelope.
            deployment_height,              # Deployment altitude of the envelope.
            margin_height,                  # Margin for the pressure altitude.
            RH,                             # Relative Humidity (0-1).   
            purity,                         # Purity of lifting gas.
            delta_P,                        # Increment in lifting gas pressure.
            delta_T,                        # Increment in lifting gas temperature.
            gas_constant,                   # Gas constant for the gas filled in the aerostat.
            inflation_fraction_oper=0.9,    # Inflation Fraction at operation.
            lobe_number=1,                  # Lobe number
            e=0, f=0, g=0                   # Lobe offsets
    ):
        P_dep, T_dep = get_atmospheric_properties(deployment_height)
        P_op,  T_op  = get_atmospheric_properties(operational_height)

        self.inflation_fraction_deploy = inflation_fraction_oper * (P_op + delta_P) / (P_dep + delta_P) * (T_dep + delta_T) / (T_op + delta_T)
        self.inflation_fraction_oper = inflation_fraction_oper
        self.inflation_fraction_factor = inflation_fraction_oper * (P_op + delta_P) / (T_op + delta_T)
        self.delta_P = delta_P
        self.delta_T = delta_T
        self.deployment_altitude = deployment_height
        self.operational_altitude = operational_height
        self.pressure_altitude = margin_height + operational_height
        self.envelope = envelope
        self.gas_properties = (RH, purity, delta_P, delta_T, gas_constant, inflation_fraction_oper)
        self.lobe_number = lobe_number
        self.multi_lobe_distances = (e, f, g)
        self.skin_density = skin_density
        self.additional_mass = additional_mass

    def initialise_from_operational_altitude (self, volume_bounds, target_lift=0):
        envelope, convergence = self.get_envelope_from_target(target_lift, self.operational_altitude, volume_bounds)
        self.envelope = envelope
        return envelope, convergence

    def get_envelope_from_target (self, target_lift, target_altitude, volume_bounds):
        envelope = self.envelope.copy()
        lobe_number = self.lobe_number
        e, f, g = self.multi_lobe_distances
        skin_density = self.skin_density
        additional_mass = self.additional_mass
        min_volume, max_volume = volume_bounds

        min_length = envelope.set_volume(min_volume or 1e-3, lobe_number, e, f, g).length
        max_length = envelope.set_volume(max_volume, lobe_number, e, f, g).length

        # TODO: Include fin mass
        if lobe_number == 1:
            def func (l):
                envelope.set_length(l)
                volume_iter = envelope.volume()
                mass_iter = skin_density * envelope.surface_area() + additional_mass
                l = get_net_lift(volume_iter, mass_iter, target_altitude, *self.gas_properties)
                return abs(l - target_lift)
        elif lobe_number == 2:
            def func (l):
                envelope.set_length(l)
                volume_iter = envelope.volume_bilobe(f)
                mass_iter = skin_density * envelope.surface_area_bilobe(f) + additional_mass
                l = get_net_lift(volume_iter, mass_iter, target_altitude, *self.gas_properties)
                return abs(l - target_lift)
        else:
            def func (l):
                envelope.set_length(l)
                volume_iter = envelope.volume_trilobe(e, f, g)
                mass_iter = skin_density * envelope.surface_area_trilobe(e, f, g) + additional_mass
                l = get_net_lift(volume_iter, mass_iter, target_altitude, *self.gas_properties)
                return abs(l - target_lift)

        sol = minimize_scalar(func, bounds=(min_length, max_length), method='bounded', options={'xatol': 1e-8})
        convergence = (sol.fun - target_lift) / max(target_lift, 1e-3) * 100

        return envelope, convergence

    def get_properties (self, n=None):
        # If number of points to be taken for altitude is not given, it would be assumed to be taken for every 100m.
        if n is None:
            n = int((self.pressure_altitude - self.deployment_altitude) / 100)

        h = np.linspace(self.deployment_altitude, self.pressure_altitude, n)
        L = np.zeros_like(h)

        # FIX: Correctly extract offsets from the tuple defined in __init__
        e, f, g = self.multi_lobe_distances

        if self.lobe_number == 1:
            volume = self.envelope.volume()
            surface_area = self.envelope.surface_area()
        elif self.lobe_number == 2:
            volume = self.envelope.volume_bilobe(f)
            surface_area = self.envelope.surface_area_bilobe(f)
        else:
            volume = self.envelope.volume_trilobe(e, f, g)
            surface_area = self.envelope.surface_area_trilobe(e, f, g)

        total_mass = self.skin_density * surface_area + self.additional_mass

        RH, purity, delta_P, delta_T, gas_constant, _ = self.gas_properties
        P, T = get_atmospheric_properties(h)
        e_vap = get_vapour_pressure(T, RH)

        # Inflation fraction varying with altitude.
        I = self.inflation_fraction_factor * (T + self.delta_T) / (P + self.delta_P)
        I = np.clip(I, 0, 1)

        # Total ballonet volume varying with altitude.
        BV = (1 - I) * volume

        rho_lg = purity * (P + delta_P) / (gas_constant * (T + delta_T))
        rho_ba = P/(287*T)

        # Gross static lift
        Lg = K * volume * (P - (1-RDWV)*e_vap) / T

        # Net static lift
        Ln = Lg - (rho_lg * I * volume + rho_ba * BV + total_mass) * g

        return h, Ln, Lg, I, BV

# Testing

# from geometry_handler import GertlerEnvelope, STANDARD_ENVELOPES
# aerostat = AerostatHull(
#         GertlerEnvelope.from_parameters(STANDARD_ENVELOPES["NPL"], 1),
#         additional_mass=150+70,
#         skin_density=.75,
#         operational_height=4500,
#         deployment_height=0,
#         margin_height=500,
#         RH=.7,
#         purity=.97,
#         delta_P=500,
#         delta_T=5,
#         gas_constant=2077,
#         lobe_number=1,
#         e=0,
#         f=0,
#         g=0
# )

# envv, conv = aerostat.initialise_from_operational_altitude([0, 50000])
# print(aerostat.get_properties())
# print(f"Aerostat length: {envv.length} {conv}")
