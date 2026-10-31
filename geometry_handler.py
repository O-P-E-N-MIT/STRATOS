# Input Variables for Gertler Envelope Generation
#
# m1 = Point of maximum thickness
# r0 = Nose radius at the front of the airship
# r1 = Tail radius at the back of the airship
# cp = Prismatic coefficient
# l2d = Length to diameter ratio
#
# Input Variables for a symmetric NACA 4 digit Airfoil Generation
#
# t = Maximum thickness as percentage of chord

import numpy as np

from scipy.integrate import dblquad, simpson, quad

try:
    from shapely.geometry import Point, box
    from shapely.affinity import scale
    from shapely.ops import unary_union
except ImportError:
    print("Shapely not found. Trilobe volume calculations will be mocked.")

try:
    from scipy.optimize import fsolve
except ImportError:
    print("Scipy not found. Volumetric generation and Aerostatic optimisation will be mocked.")

# Standard preset envelopes.
STANDARD_ENVELOPES = {
    "Sphere":       (0.5000, 0.5000, 0.5000, 0.6670, 1.0000),
    "GNVR":         (0.4143, 0.5999, 0.1762, 0.6163, 3.0500),
    "ZHIYUAN-1":    (0.4193, 0.3306, 0.2500, 0.6489, 3.2592),
    "Wang":         (0.4040, 0.6000, 0.1000, 0.6100, 3.8540),
    "NPL":          (0.4319, 0.5886, 0.4248, 0.6667, 4.0000),
    "LOTTE":        (0.4502, 0.5759, 0.1000, 0.5170, 3.9020),
    "Garg":         (0.5001, 0.4616, 0.4601, 0.7000, 3.2093),
    "Ellipsoid":    (0.5000, 0.5000, 0.5001, 0.6667, 4.9999),
    "SkyShip":      (0.4093, 0.6522, 0.1000, 0.6131, 3.8692),
    "Custom":       (0.4150, 0.6000, 0.1800, 0.6150, 3.0440),
}

# A way to store the coefficients in the memory instead of computing it for the same parameters every time.
GERTLER_COEFFICIENTS = {}

# Resolution for shapely approximation.
RES = 128

# A general Envelope object to extend it to multiple shapes.
class Envelope:

    # Initiating the basic envelope class.
    def __init__(self, length, diameter, n):
        self.length = length
        self.diameter = diameter
        self.n = int(n)

    # Returns the coordinates of points on the envelope which intercepts the trailing edge of a fin and the necessary intercept offset.
    def get_chord_intercept (self, x, rc):
        chord_length = 0
        h = self.length / self.n
        y = self.at(x)

        x1 = x
        y1 = y
        
        X = []
        Y = []

        while chord_length < rc:
            x1 += h
            y1 = self.at(x1)

            X.append(x1)
            Y.append(y1)

            chord_length = ((x-x1)**2 + (y-y1)**2)**0.5

        # If the final chord length is smaller than the root chord, it means the axial offset is too high for a fin to be placed on the hull.
        if chord_length < rc:
            raise Exception("GeometryHandlerError: Unable to find the trailing edge intercept for the given parameters.")
        
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # Points of the root chord of the fin
        Y_RC = y + ((y1 - y)/(x1 - x)) * (X - x)
        
        # Finding the necessary offset for which the fin has to be brought closer to the axis so that the root chord
        # completely intercepts with the envelope.
        intercept_offset = np.max(Y_RC - Y)
        
        return x1, y1, intercept_offset
    
    # Set the length of the envelope
    def set_length (self, l):
        self.diameter = l * self.diameter / self.length
        self.length = l

    # Sets the length of the envelope for the given volume.
    #
    # volume = Volume of the envelope
    # lobe_number = Number of lobes 
    # e, f, g = Multi lobe distances
    #
    # NOTE: The following code assumes all the lobes are of same shape and length.
    def set_volume (self, volume, lobe_number=1, e=0, f=0, g=0):
        # Calculation of length using V = pi * D^2 * L * cp/4 for a monolobe. In case of multilobe, the formula for volume is 
        # complicated and scipy's fsolve is used to compute the required length for the given volume. 
        # 
        # For initial estimate of length of multi lobes, it is assumed there are no intersections between the lobe. This gives
        # us a good initial estimate making it more faster for us to arrive at the actual value.
        # 
        # Volume = Number of lobes * pi * D^2 * L_estimate * cp/4
        l2d = self.length / self.diameter
        initial_length = ((4 * volume * l2d**2) / (np.pi * self.cp * lobe_number))**(1/3)
        self.set_length(initial_length)

        if lobe_number == 1:
            # In case of monolobe, the initial length estimate is exactly the length of monolobe.
            return self
        
        # Different function is used in different case of number of lobes here. If we were to put this if condition within one function, 
        # it will be computationally inefficient.
        elif lobe_number == 2:
            # In case of bilobe design,
            def func (l):
                # Update the new length to the envelope.
                self.set_length(l[0])
                # New estimate of the volume from the new length.
                volume_iter = self.volume_bilobe(f)
                # Returns a factor indicating how much did the volume change with the new length.
                # With this factor approaching 0, the length will approach the required length accordingly.
                return (volume - volume_iter) / volume  
        else:
            # In case of trilobe design,
            def func (l):
                self.set_length(l[0])
                volume_iter = self.volume_trilobe(e, f, g)
                return (volume - volume_iter) / volume
                
        fsolve(func, initial_length)
        return self

    # Returns a copy of the envelope.
    def copy (self):
        return type(self)(self.coeffs, self.length, self.diameter, self.n)

    # Returns petal coordinates
    def petal_coordinates (self, petal_number, nc = 150):
        delta_phi = 2 * np.pi / petal_number
        phi_half_width = delta_phi / 2.0
        phi = np.linspace(-phi_half_width, phi_half_width, nc)

        return [(x, r * phi[j]) for x, r in self.points() for j in range(nc)]
    
    # Returns the volume of the envelope (mono lobe).
    def volume (self):
        X, R = self.points(tuples=False)
        return np.trapezoid(np.pi * R**2, X)
    
    # Returns the surface area of the envelope (mono lobe).
    def surface_area (self):
        X, R = self.points(tuples=False)
        return np.trapezoid(2 * np.pi * R, X)
    
    # Returns the side projected area of the envelope (mono lobe and bi lobe).
    def side_projected_area (self):
        X, R = self.points(tuples=False)
        return 2 * np.trapezoid(R, X)
    
    # Returns center of volume of the envelope.
    def cv (self):
        X, R = self.points(tuples=False)
        return np.trapezoid(X * np.pi * R**2, X) / self.volume(), 0
    
    # Returns the volume of a bilobed envelope.
    #
    # f = Distance of the extreme lobe from X axis.
    def volume_bilobe (self, f):
        # If there is no distance between lobes, both the lobes coincide.
        if f == 0:
            return self.volume()

        X, R = self.points(tuples=False)    # Array of coordinates of the envelope
        A = np.zeros(self.n)                # Array of intersections of cross section

        # Calculation intersecting cross sectional area
        r = R[f < R]
        A[f < R] =  2 * (r**2 * np.acos(f/r) - f * np.sqrt(r**2 - f**2))

        # 2 * (Volume of the lobes) - Intersection of both the lobes
        return 2 * self.volume() - np.trapezoid(A, X)
    
    # Returns the surface area of bilobe.
    def surface_area_bilobe (self, f):
        X, R = self.points(tuples=False) 
        
        P = 4 * R
        P[R >= f] *= np.pi - np.arccos(f / (2 * R[R >= f]))
        P[R < f] *= np.pi

        return np.trapezoid(P, X)
    
    # Returns the top projected area of a bilobe.
    def top_projected_area_bilobe (self, f):
        X, R = self.points(tuples=False)
        L = R + np.minimum(R, f)
        return 2 * np.trapezoid(L, X)
    
    # Returns the center of volume of bilobe.
    def cv_bilobe (self, f):
        if f == 0: return self.cv()

        X, R = self.points(tuples=False)    
        A = 2 * np.pi * R**2

        r = R[f < R]
        A[f < R] -= 2 * (r**2 * np.acos(f/r) - f * np.sqrt(r**2 - f**2))

        return np.trapezoid(X * A, X) / np.trapezoid(A, X), 0
    
    # Returns the volume of a trilobed envelope.
    #
    # e = Distance of central lobe from Y axis
    # f = Distance of extreme lobe from X axis
    # g = Distance of central lobe from Z axis
    #
    # NOTE: For trilobe volume calculation, shapely module is used. Try to find some other way to compute
    # it other way which is more accurate and faster. Shapely calculations are not accurate but the error is
    # often negligible. There are no closed forms for trisection calculations.
    #
    # TODO: Test this properly as I only tried only with few specific cases. Later, optimise the code.
    def volume_trilobe (self, e, f, g, central_lobe = None):
        # If central lobe is not provided, itself is taken as central lobe.
        X, R, R2 = get_trilobe_axis(self, central_lobe or self, e)

        # Array of cross section areas along the axis.
        A = [unary_union([Point(-f, 0).buffer(R[i]), Point(f, 0).buffer(R[i]), Point(0, g).buffer(R2[i])]).area for i in range(len(X))]

        # Integrating the area of cross section to get the volume.
        return np.trapezoid(A, X)
    
    # Returns the surface area of a trilobe design.
    def surface_area_trilobe (self, e, f, g, central_lobe = None):
        X, R, R2 = get_trilobe_axis(self, central_lobe or self, e)

        # Array of perimeters of cross section
        P = [unary_union([Point(-f, 0).buffer(R[i]), Point(f, 0).buffer(R[i]), Point(0, g).buffer(R2[i])]).length for i in range(len(X))]

        # Integrating the perimeters of the cross section to get the area.
        return np.trapezoid(P, X)
    
    # Returns the side projected area of a trilobe design.
    def side_projected_area_trilobe (self, e, f, g, central_lobe = None):
        X, R, R2 = get_trilobe_axis(self, central_lobe or self, e)

        Y_TOP = np.minimum(R, g + R2)
        Y_BOTTOM = np.maximum(-R, g - R2)
        L = 2 * (R + R2) - np.maximum(0, Y_TOP - Y_BOTTOM)

        return np.trapezoid(L, X)
    
    # Returns the top projected area of a trilobe design.
    def top_projected_area_trilobe (self, e, f, g, central_lobe = None):
        X, R, R2 = get_trilobe_axis(self, central_lobe or self, e)

        Y_TOP = np.minimum(R + f, R2)
        Y_BOTTOM = np.maximum(0, f - R)
        L = R + R2 + np.minimum(R, f) + np.maximum(0, Y_TOP - Y_BOTTOM)

        return 2 * np.trapezoid(L, X)
    
    # Returns the center of volume for a trilobe design.
    def cv_trilobe (self, e, f, g, central_lobe = None):
        X, R, R2 = get_trilobe_axis(self, central_lobe or self, e)
        Y = np.zeros_like(X)        # Center of area in vertical direction due to lobe offsets
        A = np.zeros_like(X)        # Array of cross sections

        for i in range(len(X)):
            union = unary_union([Point(-f, 0).buffer(R[i]), Point(f, 0).buffer(R[i]), Point(0, g).buffer(R2[i])])

            A[i] = union.area
            if A[i]: 
                Y[i] = union.centroid.y
        
        V = np.trapezoid(A, X)
        return np.trapezoid(X * A, X) / V, np.trapezoid(Y * A, X) / V
    
    @classmethod
    def from_parameters_volume (model, params, volume, n=100, lobe_number=1, e=0, f=0, g=0):
        envelope = model.from_parameters(params, 1, n)
        envelope.set_volume(volume, lobe_number, e, f, g)
        return envelope

class DragonDreamEnvelope(Envelope):

    def __init__(self, length, width, height, bottom_flatness, n=100):
        self.length = length
        self.width = width
        self.height = height
        self.bottom_flatness = bottom_flatness # 0 to 1 (1 = fully flat at axis)
        
        # For aerostat calculations, the maximum vertical dimension (height) is taken as 
        super().__init__(length, height, n)

    # Returns a DragonDreamEnvelope from the parameters.
    #
    # params = (W, H, f)
    # length = Length of the envelope
    # n = Number of points to be generated along the length
    def from_parameters (params, length, n=100):
        return DragonDreamEnvelope(length, params[0], params[1], params[2], n)
    
    def copy (self):
        return type(self)(self.length, self.width, self.height, self.bottom_flatness, self.n)
    
    # Set the length of the envelope
    def set_length (self, l):
        self.width = l * self.width / self.length
        self.height = l * self.height / self.length
        self.diameter = self.height
        self.length = l

    def at(self, x):
        # Returns the Z-height of the top ellipsoid at a given X for fin/wing placement.
        # X is from 0 to L. Center is at L/2.
        x_local = x - (self.length / 2.0)
        a = self.length / 2.0
        c = self.height / 2.0

        # Ellipse equation: (x/a)^2 + (z/c)^2 = 1 -> z = c * sqrt(1 - (x/a)^2)
        if abs(x_local) > a:
            return 0.0
        return c * np.sqrt(1 - (x_local / a)**2)
    
    # This is only used to guess initial length in case of an volumetric input.
    @property
    def cp(self):
        return 0.5 * self.width / self.height * (2/3 - (self.bottom_flatness - 1) + (self.bottom_flatness - 1)**3 / 3)

    def volume(self):
        print(self.length * self.height * self.width)
        return np.pi/8 * (self.length * self.width * self.height) * (2/3 - (self.bottom_flatness - 1) + (self.bottom_flatness - 1)**3 / 3)

    def surface_area(self):
        a = self.length / 2
        b = self.width / 2
        c = self.height / 2
        f = self.bottom_flatness

        S_flat = (np.pi * self.length * self.width / 4) * f * (2 - f)

        u_upper = np.arccos(f - 1)

        S_curved, _ = dblquad(
            lambda u, v: 
                np.sin(u) * np.sqrt(
                    (b * c * np.sin(u) * np.cos(v))**2
                    + (a * c * np.sin(u) * np.sin(v))**2
                    + (a * b * np.cos(u))**2
                ),

            0, 2*np.pi, lambda _: 0, lambda _: u_upper)

        return S_flat + S_curved
    
    def cv(self):
        k = self.bottom_flatness - 1
        return 0, (self.height / 8) * (1 - k**2)**2 / (2/3 - k + k**3 / 3)

    def side_projected_area(self):
        k = self.bottom_flatness - 1
        return 0.25 * (self.length * self.height) * (np.pi/2 - np.arcsin(k) - k * np.sqrt(1 - k**2))
    
    def volume_bilobe(self, f):
        if f > self.width:
            return 2 * self.volume()
        
        a = self.length / 2
        b = self.width / 2
        c = self.height / 2
        bf = self.bottom_flatness
        z_min = -c * (1 - bf)
    
        def z_lower(y):
            return max(z_min, -c * np.sqrt(1 - (y / b)**2))
    
        def z_upper(y):
            return c * np.sqrt(1 - (y / b)**2)
    
        def x_val(z, y):
            val = 1 - (y / b)**2 - (z / c)**2
            return a * np.sqrt(val) if val > 0 else 0.0
    
        return 2 * self.volume() - 4 * dblquad(x_val, f/2, b, z_lower, z_upper)[0]
    
    def surface_area_bilobe(self, f):
        if f > self.width:
            return 2 * self.surface_area()
        
        a = self.length / 2
        b = self.width / 2
        c = self.height / 2
        bf = self.bottom_flatness

        def surface_integrand(u, v):
            return np.sin(u) * np.sqrt(
                (b * c * np.sin(u) * np.cos(v))**2
                + (a * c * np.sin(u) * np.sin(v))**2
                + (a * b * np.cos(u))**2
            )
    
        def v_lower_hidden(u):
            ratio = f / (2 * b * np.sin(u)) if np.sin(u) != 0 else 1.0
            return np.arcsin(ratio) if ratio < 1.0 else 0.0
    
        def v_upper_hidden(u):
            ratio = f / (2 * b * np.sin(u)) if np.sin(u) != 0 else 1.0
            return np.pi - np.arcsin(ratio) if ratio < 1.0 else 0.0
        
        u_min_overlap = np.arcsin(f / (2 * b))
        u_max_overlap = np.arccos(bf - 1)
    
        S_hidden_curved = dblquad(
            lambda v, u: surface_integrand(u, v), 
            u_min_overlap, u_max_overlap,
            v_lower_hidden, v_upper_hidden
        )[0] if u_min_overlap < u_max_overlap else 0
    
        r_flat = np.sqrt(bf * (2 - bf))
        a_flat = a * r_flat
        b_flat = b * r_flat
    
        S_hidden_flat = quad(
            lambda y: 2 * a_flat * np.sqrt(max(1 - (y / b_flat)**2, 0)), 
            f/2, b_flat
        )[0] if f / 2 < b_flat else 0
    
        return 2 * self.surface_area() - 2 * (S_hidden_curved + S_hidden_flat)
    
    def top_projected_area_bilobe(self, f):
        if f > self.width:
            return np.pi/2 * self.length * self.width
        
        a = self.length / 2
        b = self.width / 2
        term_top = f / (2 * b)

        return 2 * (np.pi * a * b) - 2 * a * b * (np.pi/2 - np.arcsin(term_top) - term_top * np.sqrt(1 - term_top**2))
    
    def cv_bilobe(self, f):
        if f > self.width:
            return self.cv()
        
        a = self.length / 2
        b = self.width / 2
        c = self.height / 2
        bf = self.bottom_flatness
        z_min = -c * (1 - bf)
    
        M_z_single = (np.pi * a * b * c**2 / 4) * (1 - (1 - bf)**2)**2
    
        def z_lower(y):
            return max(z_min, -c * np.sqrt(1 - (y / b)**2))
    
        def z_upper(y):
            return c * np.sqrt(1 - (y / b)**2)
    
        def x_val(z, y):
            val = 1 - (y / b)**2 - (z / c)**2
            return a * np.sqrt(val) if val > 0 else 0.0
    
        M_z_overlap_quarter, _ = dblquad(lambda z, y: z * x_val(z, y), f / 2, b, z_lower, z_upper)

        return 0, (2 * M_z_single - 4 * M_z_overlap_quarter) / self.volume_bilobe(f)
    
    def top_projected_area_trilobe(self, e, f, g, central_lobe=None):
        a = self.length / 2
        b = self.width / 2
            
        return unary_union([
            scale(Point(0, 0).buffer(1, resolution=RES), xfact=a, yfact=b),
            scale(Point(e, f).buffer(1, resolution=RES), xfact=a, yfact=b),
            scale(Point(e, -f).buffer(1, resolution=RES), xfact=a, yfact=b)
        ]).area
    
    def side_projected_area_trilobe(self, e, f, g, central_lobe=None):
        a = self.length / 2
        c = self.height / 2
        z_cut = -c * (1 - self.bottom_flatness)

        return unary_union([
            scale(Point(0, g).buffer(1, resolution=RES), xfact=a, yfact=c)
                .intersection(box(0 - 2*a, g + z_cut, 0 + 2*a, g + 2*c)),
            scale(Point(e, 0).buffer(1, resolution=RES), xfact=a, yfact=c)
                .intersection(box(e - 2*a, 0 + z_cut, e + 2*a, 0 + 2*c))
        ]).area
    
    def volume_trilobe(self, e, f, g, central_lobe=None):
        a = self.length / 2
        b = self.width / 2
        c = self.height / 2
        f_flat = self.bottom_flatness
    
        z_cut_local = -c * (1 - f_flat)
        centers = [(0.0, 0.0, g), (e, f, 0.0), (e, -f, 0.0)]
    
        z_mins = [cz + z_cut_local for _, _, cz in centers]
        z_maxs = [cz + c for _, _, cz in centers]
        global_z_min = min(z_mins)
        global_z_max = max(z_maxs)
    
        z_array = np.linspace(global_z_min, global_z_max, 1000)
        areas = []
    
        for z in z_array:
            slice_ellipses = []
            for cx, cy, cz in centers:
                if cz + z_cut_local <= z <= cz + c:
                    dz = z - cz
                    val = 1 - (dz / c) ** 2
                    if val > 0:
                        local_a = a * np.sqrt(val)
                        local_b = b * np.sqrt(val)
                        ellipse = scale(Point(cx, cy).buffer(1.0, resolution=RES), xfact=local_a, yfact=local_b)
                        slice_ellipses.append(ellipse)
            areas.append(unary_union(slice_ellipses).area if slice_ellipses else 0.0)
    
        return simpson(y=np.array(areas), x=z_array)
    
    def surface_area_trilobe(self, e, f, g, central_lobe=None):
        a = self.length / 2
        b = self.width / 2
        c = self.height / 2
        f_flat = self.bottom_flatness
    
        z_cut_local = -c * (1 - f_flat)
        centers = [(0.0, 0.0, g), (e, f, 0.0), (e, -f, 0.0)]
    
        z_mins = [cz + z_cut_local for _, _, cz in centers]
        z_maxs = [cz + c for _, _, cz in centers]
        global_z_min = min(z_mins)
        global_z_max = max(z_maxs)
    
        z_array = np.linspace(global_z_min, global_z_max, 1000)
        perimeters = []
    
        for z in z_array:
            slice_ellipses = []
            for cx, cy, cz in centers:
                if cz + z_cut_local <= z <= cz + c:
                    dz = z - cz
                    val = 1 - (dz / c) ** 2
                    if val > 0:
                        local_a = a * np.sqrt(val)
                        local_b = b * np.sqrt(val)
                        ellipse = scale(Point(cx, cy).buffer(1.0, resolution=RES), xfact=local_a, yfact=local_b)
                        slice_ellipses.append(ellipse)
            perimeters.append(unary_union(slice_ellipses).length if slice_ellipses else 0.0)
    
        S_curved = simpson(y=np.array(perimeters), x=z_array)
    
        bottom_ellipses = []
        for cx, cy, cz in centers:
            if np.isclose(cz + z_cut_local, global_z_min, atol=1e-5):
                r = np.sqrt(f_flat * (2 - f_flat))
                ellipse = scale(Point(cx, cy).buffer(1.0, resolution=RES), xfact=a * r, yfact=b * r)
                bottom_ellipses.append(ellipse)
    
        S_flat = unary_union(bottom_ellipses).area if bottom_ellipses else 0.0
    
        return S_curved + S_flat
    
    def cv_trilobe(self, e, f, g, central_lobe=None):
        a = self.length / 2
        b = self.width / 2
        c = self.height / 2
        f_flat = self.bottom_flatness
    
        z_cut_local = -c * (1 - f_flat)
        centers = [(0.0, 0.0, g), (e, f, 0.0), (e, -f, 0.0)]
    
        z_mins = [cz + z_cut_local for _, _, cz in centers]
        z_maxs = [cz + c for _, _, cz in centers]
        global_z_min = min(z_mins)
        global_z_max = max(z_maxs)
    
        z_array = np.linspace(global_z_min, global_z_max, 1000)
        areas = []
        x_centroids = []
    
        for z in z_array:
            slice_ellipses = []
            for cx, cy, cz in centers:
                if cz + z_cut_local <= z <= cz + c:
                    dz = z - cz
                    val = 1 - (dz / c) ** 2
                    if val > 0:
                        local_a = a * np.sqrt(val)
                        local_b = b * np.sqrt(val)
                        ellipse = scale(Point(cx, cy).buffer(1.0, resolution=RES), xfact=local_a, yfact=local_b)
                        slice_ellipses.append(ellipse)
            if slice_ellipses:
                union_poly = unary_union(slice_ellipses)
                areas.append(union_poly.area)
                x_centroids.append(union_poly.centroid.x)
            else:
                areas.append(0.0)
                x_centroids.append(0.0)
    
        areas = np.array(areas)
        x_centroids = np.array(x_centroids)
    
        V = simpson(y=areas, x=z_array)
        X_c = simpson(y=x_centroids * areas, x=z_array) / V if V > 0 else 0.0
        Z_c = simpson(y=z_array * areas, x=z_array) / V if V > 0 else 0.0
    
        return X_c, Z_c
    
class GertlerEnvelope(Envelope):

    # Input variables for Gertler Envelope
    #
    # coeffs = Coefficients of the Gertler polynomial
    # length = Length of the envelope
    # diameter = Maximum diameter of the envelope
    # n = Number of points to be generated along the length
    #
    # NOTE: Coefficients for Gertler Enevelope is to be of the following form. 
    # a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6
    def __init__ (self, coeffs, length, diameter, n):
        self.coeffs = list(coeffs)
        super().__init__(length, diameter, n)

    # Get the radius of the envelope at a particular axial position x
    def at (self, x):
        x = x / self.length
        y = self.diameter * max(self.coeffs[0]*x + self.coeffs[1]*x**2 + self.coeffs[2]*x**3 + self.coeffs[3]*x**4 + self.coeffs[4]*x**5 + self.coeffs[5]*x**6, 0)**0.5
        return y
    
    # Returns an array of points representing the Gertler envelope.
    #
    # NOTE: These points are not linearly scalable with length like NACA airfoils.
    # NOTE: I have made use of numpy array here but an iterator function would be memory efficient I guess?
    def points (self, truncation = 0, tuples = True):
        X = np.linspace(0, 1 - truncation, self.n)
        R = np.polyval(self.ordered_coeffs, X)
        R[R < 0] = 0

        # In case if there is no truncation, the final point must lie on the axis.
        if not truncation:
            R[-1] = 0 

        R = self.diameter * np.sqrt(R)
        X = self.length * X

        # Zipping elements maybe a bad idea.
        return zip(X, R) if tuples else (X, R)
    
    # Returns the volume of the envelope (mono lobe).
    # Method overriden from general envelope by making use of the properties of Gertler Envelope.
    def volume (self):
        # pi x D^2 x L x cp/4
        return np.pi * (self.diameter**2) * self.length * np.dot(self.coeffs, [1/2, 1/3, 1/4, 1/5, 1/6, 1/7])

    # Returns the coefficients in an ordered way numpy usually accepts.
    @property
    def ordered_coeffs (self):
        return self.coeffs[::-1] + [0]
    
    # Returns the prismatic coefficient of the envelope.
    @property
    def cp (self):
        return np.dot(self.coeffs, [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]) * 4

    # Returns the coefficients of the Gertler polynomial from standard parameters.
    def get_coefficients (params):
        # If the coefficients for the given parameters have been computed already, send that.
        if params in GERTLER_COEFFICIENTS:
            return GERTLER_COEFFICIENTS[params]
        
        (m, r0, r1, cp, _) = params

        A = np.array([   
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [m, m**2, m**3, m**4, m**5, m**6],
            [1, 2*m, 3*m**2, 4*m**3, 5*m**4, 6*m**5],
            [1, 2, 3, 4, 5, 6],
            [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]
        ], float)

        B = np.array([2*r0, 0, 1/4, 0, -2*r1, 1/4*cp], float).T
        
        try:
            X = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            print("[ERROR] Coefficient calculation for gertler envelope failed.")
            return np.zeros(6)
        
        GERTLER_COEFFICIENTS[params] = X
        
        return np.round(X, 4)
    
    # Returns a GertlerEnvelope from standard parameters.
    #
    # params = (m1, r0, r1, cp, l2d)
    # length = Length of the envelope
    # n = Number of points to be generated along the length
    def from_parameters (params, length, n=100):
        coeffs = GertlerEnvelope.get_coefficients(params)
        diameter = length / params[4]
        return GertlerEnvelope(coeffs, length, diameter, n)
    
    def copy (self):
        return type(self)(self.coeffs, self.length, self.diameter, self.n)
    
class NACAEnvelope(Envelope):

    # Prismatic coefficient of a NACA envelope is constant irrespective of the parameters.
    cp = 0.5477

    def __init__(self, length, diameter, n):
        super().__init__(length, diameter, n)

    def at (self, x):
        x = x / self.length
        y = 5 * self.diameter * (0.2969 * x**0.5 - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
        return y
    
    # Returns an array of points representing the envelope.
    def points (self, truncation = 0, tuples = True):
        X = np.linspace(0, 1 - truncation, self.n)
        R = 5 * self.diameter * (0.2969 * X**0.5 - 0.1260 * X - 0.3516 * X**2 + 0.2843 * X**3 - 0.1036 * X**4)
        R[R < 0] = 0

        # In case if there is no truncation, the final point must lie on the axis.
        if not truncation:
            R[-1] = 0 

        X = self.length * X

        # Zipping elements maybe a bad idea.
        return zip(X, R) if tuples else (X, R)

    # Returns a NACAEnvelope from the naca parameters.
    #
    # params = (l2d)
    # length = Length of the envelope
    # n = Number of points to be generated along the length
    def from_parameters (params, length, n=100):
        diameter = length / params[0]
        return NACAEnvelope(length, diameter, n)
    
    def copy (self):
        return type(self)(self.length, self.diameter, self.n)

# Gets a common trilobe axis for both extreme and central lobe.
def get_trilobe_axis (extreme_lobe, central_lobe, e):
    # Determining how long should the discretized X axis should be taken for calculation.
    last = max(extreme_lobe.length, e + central_lobe.length)
    # NOTE: Number of elements taking the same as the main envelope is not a good idea.
    X = np.linspace(0, last, extreme_lobe.n)

    # Getting the indices of position of start of central lobe, end of central lobe, end of extreme lobe in the
    # discretized X axis. This may not be the best way to do this.
    extreme_lobe_end_index = np.argmin(np.abs(X - extreme_lobe.length))
    central_lobe_start_index = np.argmin(np.abs(X - e))
    central_lobe_end_index = np.argmin(np.abs(X - (e + central_lobe.length)))

    # Due to discretization of X axis, the extreme lobe length and distance of central lobe from Y axis will be
    # different than the ideal values specified. Higher the value of n, closer will be to their ideal values.
    # TODO: Again, this may not be the best way to do it.
    extreme_lobe_length = X[extreme_lobe_end_index] or 1e-8 # Ideal value: self.envelope
    central_lobe_start = X[central_lobe_start_index]        # Ideal value: e

    # Radius of extreme lobe along the X axis
    # NOTE: This maybe a crude way to do this wihout using .points()
    R = np.zeros_like(X)
    R[:extreme_lobe_end_index+1] = np.polyval(extreme_lobe.ordered_coeffs, X[:extreme_lobe_end_index+1] / extreme_lobe_length)
    R[extreme_lobe_end_index:] = 0
    R = extreme_lobe.diameter * np.sqrt(R)

    # Radius of central lobe along the X axis.
    R2 = np.zeros_like(X)
    R2[central_lobe_start_index:] = np.polyval(central_lobe.ordered_coeffs, (X[central_lobe_start_index:] - central_lobe_start) / central_lobe.length)
    R2[central_lobe_end_index:] = 0
    R2 = central_lobe.diameter * np.sqrt(R2)

    return X, R, R2

# Length scaling for volumetric generation.
def length_scaling (volume_function, target_volume, initial_estimate):
    def f (l):
        volume = volume_function(l)
        return (target_volume - volume) / target_volume
    
    return fsolve(f, initial_estimate)[0]

class UnifiedAsymmetricEnvelope(Envelope):
    """
    Unified LTA envelope generator (Alam, SSRN 7274120).
    Represents an asymmetric hull utilizing three directional scaling factors
    and three discrete Gertler profile parameter sets.
    """
    def __init__(self, p_matrix, scales, length, n=100):
        # p_matrix: 3x4 array -> [m, r0, r1, Cp] for Lateral, Dorsal, Ventral
        # scales: [alpha, beta, gamma] -> Lateral width, 2x Upper Height, 2x Lower Height
        self.p_matrix = np.array(p_matrix)
        self.scales = np.array(scales)
        self.length = length

        # Use beta (dorsal height max) as the nominal diameter for the base class
        super().__init__(length, self.scales[1], n)

        # Solve Gertler polynomials for all 3 profiles
        self.coeffs = np.zeros((3, 6))
        for k in range(3):
            m, r0, r1, cp = self.p_matrix[k]
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [m, m**2, m**3, m**4, m**5, m**6],
                [1, 2*m, 3*m**2, 4*m**3, 5*m**4, 6*m**5],
                [1, 2, 3, 4, 5, 6],
                [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]
            ], float)
            B = np.array([2*r0, 0, 1/4, 0, -2*r1, 1/4*cp], float).T
            self.coeffs[k] = np.linalg.solve(A, B)

    def _get_radii(self, x):
        """Returns (r_lat, r_dorsal, r_ventral) for a given axial station x."""
        x_norm = np.clip(x / self.length, 0, 1)
        scalar = np.isscalar(x)
        x_norm = np.atleast_1d(x_norm)
        radii = np.zeros((x_norm.shape[0], 3))

        for k in range(3):
            ordered = np.append(self.coeffs[k][::-1], 0) # Append a0 = 0
            q = np.maximum(np.polyval(ordered, x_norm), 0)
            radii[:, k] = self.scales[k] * np.sqrt(q)

        return radii[0] if scalar else radii

    def at(self, x):
        """Returns the upper (dorsal) radius at x for fin and wing placement logic."""
        return self._get_radii(x)[1]

    def volume(self):
        x = np.linspace(0, self.length, self.n)
        r = self._get_radii(x)
        # Cross-sectional area of two half-ellipses sharing a lateral semi-axis
        A = (np.pi / 2) * r[:, 0] * (r[:, 1] + r[:, 2])
        return np.trapezoid(A, x)

    def surface_area(self):
        x = np.linspace(0, self.length, self.n)
        theta = np.linspace(-np.pi, np.pi, 121)[:-1] # Drop duplicate seam
        r = self._get_radii(x)

        TH, R_lat = np.meshgrid(theta, r[:, 0])
        _, R_top = np.meshgrid(theta, r[:, 1])
        _, R_bot = np.meshgrid(theta, r[:, 2])
        XX, _ = np.meshgrid(theta, x)

        Y = R_lat * np.cos(TH)
        Z = (R_top * (np.sin(TH) >= 0) + R_bot * (np.sin(TH) < 0)) * np.sin(TH)

        # Vectorized 3D surface area integration
        p1 = np.stack((XX[:-1, :], Y[:-1, :], Z[:-1, :]), axis=2)
        p2 = np.stack((XX[1:, :], Y[1:, :], Z[1:, :]), axis=2)
        p3 = np.roll(p2, -1, axis=1)
        p4 = np.roll(p1, -1, axis=1)

        a1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1, axis=2), axis=2)
        a2 = 0.5 * np.linalg.norm(np.cross(p3 - p1, p4 - p1, axis=2), axis=2)

        return np.sum(a1) + np.sum(a2)

    def cv(self):
        x = np.linspace(0, self.length, self.n)
        r = self._get_radii(x)
        A = (np.pi / 2) * r[:, 0] * (r[:, 1] + r[:, 2])
        vol = np.trapezoid(A, x)

        x_cb = np.trapezoid(x * A, x) / vol
        z_moment = (2/3) * r[:, 0] * (r[:, 1]**2 - r[:, 2]**2)
        z_cb = np.trapezoid(z_moment, x) / vol

        return x_cb, z_cb

    def side_projected_area(self):
        x = np.linspace(0, self.length, self.n)
        r = self._get_radii(x)
        return np.trapezoid(r[:, 1] + r[:, 2], x)

    def top_projected_area(self):
        x = np.linspace(0, self.length, self.n)
        r = self._get_radii(x)
        return 2 * np.trapezoid(r[:, 0], x)

    @classmethod
    def from_parameters(cls, params, length, n=100):
        # params unpacked as (p_matrix, scales)
        p_matrix, scales = params
        return cls(p_matrix, scales, length, n)

    def set_length(self, l):
        scale_factor = l / self.length
        self.scales = self.scales * scale_factor
        self.diameter = self.scales[1]
        self.length = l
