import os
import numpy as np
import math

from scipy.optimize import least_squares
from scipy.linalg import solve
from math import factorial

def scan_airfoil_directory(folder_path):
    """
    Scans a directory for .dat or .txt files, validating them to ensure they contain
    readable airfoil coordinates. Acts as the backend database worker.
    """
    valid_airfoils = {}
    if not os.path.exists(folder_path):
        return valid_airfoils

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.dat', '.txt')):
            continue

        filepath = os.path.join(folder_path, filename)

        # Skip empty files
        if os.path.getsize(filepath) == 0:
            continue

        # Quick validation check: Must have at least 10 valid coordinate lines
        valid_lines = 0
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        valid_lines += 1
                        if valid_lines >= 10:
                            break  # Good enough to be considered a valid geometry file
                    except ValueError:
                        continue

        if valid_lines >= 10:
            # Use the filename without extension as the display name
            display_name = os.path.splitext(filename)[0]
            valid_airfoils[display_name] = filepath

    return valid_airfoils

class BaseFitter:

    def __init__(self, filename):
        # Read x, y coordinates robustly to handle both CSV and Selig .dat formats
        self.xcoords_orig = []
        self.ycoords_orig = []

        with open(filename, 'r') as f:
            for line in f:
                # Replace commas with spaces to handle both CSV and space-delimited DAT files seamlessly
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        self.xcoords_orig.append(x)
                        self.ycoords_orig.append(y)
                    except ValueError:
                        # Skip header text or invalid lines
                        continue

        # Convert to numpy arrays
        xcoords = np.array(self.xcoords_orig)
        ycoords = np.array(self.ycoords_orig)

        self.le_index = None
        self.translation_orig = None
        self.rotation_angle_orig = None
        self.scale_factor_orig = None

        # Find leading edge
        self.le_index = np.argmin(xcoords)
        x_le = xcoords[self.le_index]
        y_le = ycoords[self.le_index]

        # Translate LE to origin
        x_translated = xcoords - x_le
        y_translated = ycoords - y_le
        self.translation_orig = (x_le, y_le)

        # Find trailing edge
        x_te = (x_translated[0] + x_translated[-1]) / 2
        y_te = (y_translated[0] + y_translated[-1]) / 2

        # Rotate to align with x-axis
        self.rotation_angle_orig = np.arctan2(y_te, x_te)
        cos_theta = np.cos(-self.rotation_angle_orig)
        sin_theta = np.sin(-self.rotation_angle_orig)

        x_rotated = x_translated * cos_theta - y_translated * sin_theta
        y_rotated = x_translated * sin_theta + y_translated * cos_theta

        # Scale to chord = 1
        chord_length = np.sqrt(x_te**2 + y_te**2)

        # Prevent division by zero if geometry is corrupted
        if chord_length == 0:
            chord_length = 1.0

        self.scale_factor_orig = chord_length

        self.x_norm = x_rotated / chord_length
        self.y_norm = y_rotated / chord_length

        # Placeholders for fitted results
        self.x_fitted_norm = None
        self.y_fitted_norm = None
        self.parameters = None

    def get_points(self, scale_factor=1, translation=(0, 0), rotation_angle=0):
        if self.y_fitted_norm is None:
            self.y_fitted_norm = self.y_norm

        if self.x_fitted_norm is None:
            self.x_fitted_norm = self.x_norm

        # Scaling the points.
        x_s = self.x_fitted_norm * scale_factor
        y_s = self.y_fitted_norm * scale_factor

        # Rotating the points.
        theta = np.deg2rad(rotation_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x_r = x_s * cos_t - y_s * sin_t
        y_r = x_s * sin_t + y_s * cos_t

        # Translating the points.
        x_final = x_r + translation[0]
        y_final = y_r + translation[1]

        return x_final, y_final

    def get_rmse_error(self):
        return np.sqrt(np.mean((self.y_norm - self._fit_build(self.x_norm, self.parameters))**2))

class CST(BaseFitter):

    def __init__(self, filename):
        super().__init__(filename)

    def _class_shape(self, w, x, N1, N2, dz):
        C = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            C[i, 0] = x[i] ** N1 * ((1 - x[i]) ** N2)

        n = w.shape[0] - 1
        K = np.zeros(n + 1)
        for i in range(n + 1):
            K[i] = factorial(n) / (factorial(i) * factorial(n - i))

        S = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            S[i, 0] = 0
            for j in range(n + 1):
                term = w[j] * K[j] * x[i] ** j * ((1 - x[i]) ** (n - j))
                S[i, 0] += term

        y = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            y[i, 0] = C[i, 0] * S[i, 0] + x[i] * dz

        return y

    def _fit_build(self, x, w, flw=5, N1=0.5, N2=1):
        wu = w[:flw - 1]
        wl = w[flw - 1:-1]
        dz = w[-1] / 2

        zerind = np.argmin(x)
        xu = x[:zerind]
        xl = x[zerind:]

        yl = self._class_shape(wl, xl, N1, N2, -dz)
        yu = self._class_shape(wu, xu, N1, N2, dz)

        y = np.concatenate((yu, yl))
        return y.flatten()

    def fit(self, n_tries=5, resolution=None):
        wu_g = [0.1, 0.1, 0.1, 0.1]
        wl_g = [-0.1, -0.1, -0.1, -0.1]
        dz_g = 0
        p0 = wu_g + wl_g + [dz_g]

        flw = len(wu_g) + 1
        fun2min = lambda w: np.abs(self._fit_build(self.x_norm, w, flw, 0.5, 1) - self.y_norm)

        results = []
        for _ in range(n_tries):
            initial_guess = p0 + np.random.uniform(-0.1, 0.1, size=len(p0))
            result = least_squares(fun2min, initial_guess, method='trf', ftol=1e-6, max_nfev=1200, diff_step=1e-6)
            results.append(result)

        best_result = min(results, key=lambda res: res.cost)
        self.parameters = best_result.x

        if resolution:
            x_half = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, resolution // 2)))
            self.x_fitted_norm = np.concatenate((x_half[::-1], x_half[1:]))
        else:
            self.x_fitted_norm = self.x_norm

        self.y_fitted_norm = self._fit_build(self.x_fitted_norm, self.parameters, flw, 0.5, 1)

class Parsec(BaseFitter):

    def __init__(self, filename):
        super().__init__(filename)
        self.initial_guesses = None

    def _analysis(self):
        xcoords, ycoords = self.x_norm, self.y_norm

        ymax, ymin = max(ycoords), min(ycoords)
        ymax_index, ymin_index = np.argmax(ycoords), np.argmin(ycoords)

        x_maxy = np.clip(xcoords[ymax_index], 0.01, 0.99)
        x_miny = np.clip(xcoords[ymin_index], 0.01, 0.99)

        te_t = ycoords[0] - ycoords[-1]
        y_te = (ycoords[0] + ycoords[-1]) / 2

        x_up, y_up = xcoords[ymax_index - 1:ymax_index + 2], ycoords[ymax_index - 1:ymax_index + 2]
        coeffs_up = np.polyfit(x_up, y_up, 2)

        x_low, y_low = xcoords[ymin_index - 1:ymin_index + 2], ycoords[ymin_index - 1:ymin_index + 2]
        coeffs_low = np.polyfit(x_low, y_low, 2)

        slope_up, slope_low = 2 * coeffs_up[0], 2 * coeffs_low[0]

        le_index = np.argmin(xcoords)
        n_le_points = min(5, le_index, len(xcoords) - le_index - 1)

        x_le_up, y_le_up = xcoords[le_index:le_index + n_le_points], ycoords[le_index:le_index + n_le_points]
        rUp = abs(1 / (2 * np.polyfit(x_le_up, y_le_up, 2)[0])) if len(x_le_up) >= 3 else 0.01

        x_le_low, y_le_low = xcoords[le_index - n_le_points:le_index + 1], ycoords[le_index - n_le_points:le_index + 1]
        rlow = abs(1 / (2 * np.polyfit(x_le_low, y_le_low, 2)[0])) if len(x_le_low) >= 3 else 0.01

        det_value = abs((xcoords[1] - xcoords[0]) * (ycoords[-2] - ycoords[-1]) -
                        (ycoords[1] - ycoords[0]) * (xcoords[-2] - xcoords[-1]))
        dot_product = ((xcoords[1] - xcoords[0]) * (xcoords[-2] - xcoords[-1]) +
                       (ycoords[1] - ycoords[0]) * (ycoords[-2] - ycoords[-1]))
        beta = math.atan2(det_value, dot_product) * 180 / math.pi

        xx, yy = np.zeros(5), np.zeros(5)
        xx[0], yy[0] = xcoords[le_index], ycoords[le_index]
        xx[1], yy[1] = (xcoords[1] + xcoords[-2]) / 2, (ycoords[1] + ycoords[-2]) / 2
        xx[2], yy[2] = (xcoords[0] + xcoords[-1]) / 2, (ycoords[0] + ycoords[-1]) / 2

        det_value2 = abs((xx[0] - xx[1]) * (yy[2] - yy[1]) - (yy[0] - yy[1]) * (xx[2] - xx[1]))
        dot_product2 = (xx[0] - xx[1]) * (xx[2] - xx[1]) + (yy[0] - yy[1]) * (yy[2] - yy[1])
        alpha = math.atan2(det_value2, dot_product2) * 180 / math.pi

        self.initial_guesses = np.array([rUp, rlow, x_maxy, ymax, slope_up, x_miny, ymin, slope_low, te_t, y_te, alpha, beta])
        return self.initial_guesses

    def _fit_build(self, x, p):
        locc = np.argmin(x)
        x_up, x_low = x[:locc + 1], x[locc:]

        def foil(x_vals, aa):
            x_safe = np.maximum(x_vals, 1e-12)
            return (aa[0] * x_safe**(1/2) + aa[1] * x_safe**(3/2) +
                    aa[2] * x_safe**(5/2) + aa[3] * x_safe**(7/2) +
                    aa[4] * x_safe**(9/2) + aa[5] * x_safe**(11/2))

        # Upper matrices
        c1, c6 = np.ones(6), np.array([1, 0, 0, 0, 0, 0])
        c2 = np.array([p[2]**(1/2), p[2]**(3/2), p[2]**(5/2), p[2]**(7/2), p[2]**(9/2), p[2]**(11/2)])
        c3 = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])
        c4 = np.array([(1/2)*p[2]**(-1/2), (3/2)*p[2]**(1/2), (5/2)*p[2]**(3/2), (7/2)*p[2]**(5/2), (9/2)*p[2]**(7/2), (11/2)*p[2]**(9/2)])
        c5 = np.array([(-1/4)*p[2]**(-3/2), (3/4)*p[2]**(-1/2), (15/4)*p[2]**(1/2), (35/4)*p[2]**(3/2), (63/4)*p[2]**(5/2), (99/4)*p[2]**(7/2)])
        Cup = np.vstack((c1, c2, c3, c4, c5, c6))

        angle_up_rad = np.deg2rad(-p[10] - p[11] / 2)
        bup = np.array([p[9] + p[8]/2, p[3], np.tan(angle_up_rad), 0, p[4], np.sqrt(2 * p[0])])

        aup = solve(Cup, bup)
        foil_up = np.real(foil(x_up, aup))

        # Lower matrices
        c7, c12 = np.ones(6), np.array([1, 0, 0, 0, 0, 0])
        c8 = np.array([p[5]**(1/2), p[5]**(3/2), p[5]**(5/2), p[5]**(7/2), p[5]**(9/2), p[5]**(11/2)])
        c9 = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])
        c10 = np.array([(1/2)*p[5]**(-1/2), (3/2)*p[5]**(1/2), (5/2)*p[5]**(3/2), (7/2)*p[5]**(5/2), (9/2)*p[5]**(7/2), (11/2)*p[5]**(9/2)])
        c11 = np.array([(-1/4)*p[5]**(-3/2), (3/4)*p[5]**(-1/2), (15/4)*p[5]**(1/2), (35/4)*p[5]**(3/2), (63/4)*p[5]**(5/2), (99/4)*p[5]**(7/2)])
        Clo = np.vstack((c7, c8, c9, c10, c11, c12))

        angle_lo_rad = np.deg2rad(-p[10] + p[11] / 2)
        blo = np.array([p[9] - p[8]/2, p[6], np.tan(angle_lo_rad), 0, p[7], -np.sqrt(2 * p[1])])

        alower = solve(Clo, blo)
        foilLow = np.real(foil(x_low, alower))

        return np.concatenate((foil_up[:-1], foilLow))

    def fun_to_min(self, p):
        try:
            p_clipped = p.copy()
            p_clipped[2] = np.clip(p_clipped[2], 0.01, 0.99)
            p_clipped[5] = np.clip(p_clipped[5], 0.01, 0.99)
            p_clipped[0] = max(p_clipped[0], 1e-6)
            p_clipped[1] = max(p_clipped[1], 1e-6)

            result = self._fit_build(self.x_norm, p_clipped) - self.y_norm
            return result if np.isfinite(result).all() else np.ones_like(self.y_norm) * 1e6
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.ones_like(self.y_norm) * 1e6

    def fit(self, resolution=None):
        if self.initial_guesses is None:
            self._analysis()

        lower_bounds = np.array([1e-6, 1e-6, 0.05, -0.5, -50, 0.05, -0.5, -50, -0.1, -0.5, -45, -45])
        upper_bounds = np.array([1.0, 1.0, 0.95, 0.5, 50, 0.95, 0.5, 50, 0.1, 0.5, 45, 45])

        p0 = np.clip(self.initial_guesses, lower_bounds, upper_bounds)

        result = least_squares(self.fun_to_min, p0, bounds=(lower_bounds, upper_bounds), method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=2000, verbose=0)

        self.parameters = result.x
        self.parameters[2] = np.clip(self.parameters[2], 0.01, 0.99)
        self.parameters[5] = np.clip(self.parameters[5], 0.01, 0.99)
        self.parameters[0] = max(self.parameters[0], 1e-6)
        self.parameters[1] = max(self.parameters[1], 1e-6)

        if resolution:
            x_half = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, resolution // 2)))
            self.x_fitted_norm = np.concatenate((x_half[::-1], x_half[1:]))
        else:
            self.x_fitted_norm = self.x_norm

        self.y_fitted_norm = self._fit_build(self.x_fitted_norm, self.parameters)

def get_naca_symmetric(thickness=12, resolution=200, scale_factor=1.0, translation=(0, 0), rotation_angle=0):
    # Cosine spacing (Add // 2 to perfectly match CST/PARSEC array lengths)
    x = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, resolution // 2)))
    t = thickness / 100.0

    # Calculate half-thickness (yt)
    yt = 5*t*(0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

    # Create standard un-transformed upper and lower surfaces.
    xu_norm = x[::-1]
    yu_norm = yt[::-1]

    # Lower goes from LE to TE, skip index 0 to avoid duplicate LE point.
    xl_norm = x[1:]
    yl_norm = -yt[1:]

    # Combine to apply matrix transformations cleanly.
    x_norm = np.concatenate((xu_norm, xl_norm))
    y_norm = np.concatenate((yu_norm, yl_norm))

    # Scale
    x_s = x_norm * scale_factor
    y_s = y_norm * scale_factor

    # Rotate
    theta = np.radians(rotation_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x_r = x_s * cos_t - y_s * sin_t
    y_r = x_s * sin_t + y_s * cos_t

    # Translate
    x_final = x_r + translation[0]
    y_final = y_r + translation[1]

    return x_final, y_final

def get_airfoil_points(thickness=None, resolution=None, filename=None, method=None, scale_factor=1.0, translation=(0, 0), rotation_angle=0, n_tries=5):
    if thickness and resolution:
        return get_naca_symmetric(thickness, resolution, scale_factor, translation, rotation_angle)
    elif filename:
        if method == 'cst':
            cst_model = CST(filename)
            cst_model.fit(n_tries, resolution)
            return cst_model.get_points(scale_factor, translation, rotation_angle)
        elif method == 'parsec':
            parsec_model = Parsec(filename)
            parsec_model.fit(resolution)
            return parsec_model.get_points(scale_factor, translation, rotation_angle)
        else:
            model = CST(filename)
            model.fit(n_tries, resolution)

            model2 = Parsec(filename)
            model2.fit(resolution)

            # If the error of cst model is high, parsec model is chosen.
            if model.get_rmse_error() > model2.get_rmse_error():
                model = model2

            return model.get_points(scale_factor, translation, rotation_angle)

    raise Exception("GeometryHandlerError: Improper parameters provided for airfoil points.")
