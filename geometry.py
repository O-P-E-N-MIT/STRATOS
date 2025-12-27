import os
import subprocess
import math
import numpy as np
from geometry_handler import GertlerEnvelope

try:
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
except ImportError:
    plt = None
    print("Warning: Matplotlib not found. Plotting functions will be mocked.")

# --- CONSTANTS ---

STANDARD_ENVELOPES = {
    "Sphere":   (0.500, 0.500,  0.500,  0.667, 1.000),
    "GNVR":     (0.4143, 0.5999, 0.1762, 0.6163, 3.0500),
    "ZHIYUAN-1":(0.4193, 0.3306, 0.2500, 0.6489, 3.2592),
    "Wang":     (0.4040, 0.6000, 0.1000, 0.6100, 3.8540),
    "NPL":      (0.4319, 0.5886, 0.4248, 0.6667, 4.0000),
    "LOTTE":    (0.4502, 0.5759, 0.1000, 0.5170, 3.902),
    "Garg": (0.5001, 0.4616, 0.4601, 0.7, 3.2093),
    "Ellipsoid": (0.5, 0.5, 0.5001, 0.6667, 4.9999),
    "SkyShip Profile": (0.409273399535807, 0.652171841644409, 0.100002973853950, 0.613129461580516, 3.86921733996695),
    "Custom":   (0.415, 0.600, 0.180, 0.615, 3.044),
}

GLOBAL_G_COEFFICIENTS = {}

NUM_POINTS_X_HIGH_RES = 150
NUM_POINTS_C_DEFAULT = 150
LOTTE_COEFFS = None

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_SCRIPT_FILE = "salome_script.py"
BREAKER = "# INPUT PARAMETERS END"
BASE_SCRIPT = open(os.path.join(DIR_PATH, BASE_SCRIPT_FILE), 'r').read().split(BREAKER, 1)[1]

def calculate_gertler_coefficients(m, r0, r1, cp):
    key = (m, r0, r1, cp)
    if key in GLOBAL_G_COEFFICIENTS: return GLOBAL_G_COEFFICIENTS[key]
    m_val = m
    A = np.array([
        [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [m_val, m_val**2, m_val**3, m_val**4, m_val**5, m_val**6],
        [1, 2*m_val, 3*m_val**2, 4*m_val**3, 5*m_val**4, 6*m_val**5],
        [1, 2, 3, 4, 5, 6], [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]
    ])
    B = np.array([2 * r0, 0, 1/4, 0, -2 * r1, 1/4 * cp])
    try:
        G = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        print("Warning: Singularity detected in Gertler matrix A. Results may be inaccurate.")
        return np.zeros(6)
    G = np.round(G * 10000) / 10000
    GLOBAL_G_COEFFICIENTS[key] = G
    return G

def gertler_profile_polynomial(x_norm, G_coeffs):
    if x_norm < 0.0 or x_norm > 1.0 or len(G_coeffs) != 6: return 0.0
    x_L = x_norm
    r_sq = (G_coeffs[0] * x_L) + (G_coeffs[1] * x_L**2) + (G_coeffs[2] * x_L**3) + \
           (G_coeffs[3] * x_L**4) + (G_coeffs[4] * x_L**5) + (G_coeffs[5] * x_L**6)
    return math.sqrt(max(0.0, r_sq))

def get_max_diameter(L, l2d):
    return L / l2d

def get_hull_radius(x_abs, L_total, D_MAX, m1, r0, r1, cp, l2d):
    G_coeffs = calculate_gertler_coefficients(m1, r0, r1, cp)
    x_norm = x_abs / L_total
    R_norm = gertler_profile_polynomial(x_norm, G_coeffs)
    return R_norm * D_MAX

def calculate_gertler_profile(params, L, x_norm):
    L_D = params["l2d"]
    m1, r0, r1, cp, _ = params["ENVELOPE_PARAMS"]
    D_MAX = get_max_diameter(L, L_D)

    R_x = np.array([get_hull_radius(x_n * L, L, D_MAX, m1, r0, r1, cp, L_D) for x_n in x_norm])
    return R_x, D_MAX

def calculate_lotte_profile(params, L, x_norm, LOTTE_COEFFS):
    return calculate_gertler_profile(params, L, x_norm)


def calculate_petal_coordinates(params, hull_length, num_petals, num_points_x=NUM_POINTS_X_HIGH_RES, num_points_c=NUM_POINTS_C_DEFAULT):
    geom_type = params["type"]
    L = hull_length

    X = np.linspace(0, L, num_points_x)
    x_norm = X / L

    if geom_type in ["gertler", "GNVR", "ZHIYUAN-1", "Wang", "NPL", "Sphere", "Custom", "LOTTE"]:
        R_x, D_val = calculate_gertler_profile(params, L, x_norm)
    elif geom_type == "lotte":
        R_x, D_val = calculate_lotte_profile(params, L, x_norm, LOTTE_COEFFS)
    else:
        print("Unknown geometry type. Cannot calculate coordinates.")
        return [], 0, 0

    if R_x is None or len(R_x) == 0:
        return [], 0, 0

    delta_phi = 2 * math.pi / num_petals
    phi_half_width = delta_phi / 2.0
    Phi = np.linspace(-phi_half_width, phi_half_width, num_points_c)

    coords_2D = []

    for i in range(num_points_x):
        r = R_x[i]
        x = X[i]

        for j in range(num_points_c):
            phi = Phi[j]
            C = r * phi

            coords_2D.append((x, C))

    return coords_2D, num_points_x, num_points_c

def write_dat_file(coords_2D, dat_filename, num_points_x, num_points_c):
    try:
        with open(dat_filename, 'w') as f:
            f.write(f"# 2D Developed Petal Coordinates\n")
            f.write(f"# X-Points: {num_points_x}, C-Points: {num_points_c}\n")
            f.write(f"# X_coordinate   C_coordinate\n")

            for x, c in coords_2D:
                f.write(f"{x:.6f}\t{c:.6f}\n")

        return f"Coordinates successfully saved to DAT file:\n{dat_filename}"

    except Exception as e:
        return f"Error generating DAT file:\n{e}"

def plot_and_save_profile(coords_2D, shape_name, dat_filename, num_points_x, num_points_c, num_petals):
    dat_result = write_dat_file(coords_2D, dat_filename, num_points_x, num_points_c)
    plot_result = ""

    if plt is None:
        plot_result = "\nMatplotlib not available. Cannot generate 2D plot."
    else:
        try:
            X_flat = [c[0] for c in coords_2D]
            C_flat = [c[1] for c in coords_2D]
            X_grid = np.array(X_flat).reshape((num_points_x, num_points_c))
            C_grid = np.array(C_flat).reshape((num_points_x, num_points_c))

            png_filename = os.path.splitext(dat_filename)[0] + "_petal.png"

            plt.figure(figsize=(10, 8))

            plt.plot(X_grid[:, 0], C_grid[:, 0], 'r-', label='Longitudinal Edge (Boundary)', linewidth=2)
            plt.plot(X_grid[:, -1], C_grid[:, -1], 'r-', linewidth=2)

            plt.plot(X_grid[:, num_points_c // 2], C_grid[:, num_points_c // 2], 'k--', label='Petal Centerline', linewidth=1)

            plt.plot(X_grid[0, :], C_grid[0, :], 'b-', label='Axial Boundary', linewidth=2)
            plt.plot(X_grid[num_points_x-1, :], C_grid[num_points_x-1, :], 'b-', linewidth=2)

            plt.title(f'2D Developed Petal: {shape_name} ({num_petals} Petals)', fontsize=16)
            plt.xlabel('Axial Position (X) [units]', fontsize=12)
            plt.ylabel('Circumferential Distance (C) [units]', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)

            handles, labels = plt.gca().get_legend_handles_labels()
            unique_labels = []
            unique_handles = []
            for h, l in zip(handles, labels):
                if l not in unique_labels:
                    unique_labels.append(l)
                    unique_handles.append(h)
            plt.legend(unique_handles, unique_labels)

            plt.axis('equal')

            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()

            plot_result = f"Plot successfully saved to:\n{png_filename}"

        except Exception as e:
            plot_result = f"Error generating plot:\n{e}"

    return f"{dat_result}\n\n{plot_result}"

class AirshipGeometry:

    def __init__(self, params, salome_path):
        self.params = params
        self.salome_path = salome_path
        self.output_directory = params["OUTPUT_DIRECTORY"]
        self.L = params["ENVELOPE_LENGTH"]
        self.l2d = params["l2d"]
        self.D_MAX = get_max_diameter(self.L, self.l2d)
        self.R_MAX = self.D_MAX / 2.0
        GLOBAL_G_COEFFICIENTS.clear()
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def _generate_salome_script(self, export_file, export_format, open_gui):
        script_filename = os.path.splitext(export_file)[0] + "_salome_script.py"
        script_path = os.path.join(self.output_directory, script_filename)

        safe_output_dir_host = os.path.normpath(self.output_directory)

        script_content = ["# INPUT PARAMETERS START"]

        if "VOLUME" in self.params:
            self.params["ENVELOPE_LENGTH"] = GertlerEnvelope.from_parameters_volume(self.params["ENVELOPE_PARAMS"], self.params["VOLUME"], self.params["ENVELOPE_RESOLUTION"], self.params["LOBE_NUMBER"], self.params["LOBE_OFFSET_X"], self.params["LOBE_OFFSET_Y"], self.params["LOBE_OFFSET_Z"]).length

        self.params.setdefault("ENVELOPE_TRUNCATION_RATIO", 0)
        self.params.setdefault("CENTRAL_LOBE_PARAMS", self.params["ENVELOPE_PARAMS"])
        self.params.setdefault("CENTRAL_LOBE_LENGTH", self.params["ENVELOPE_LENGTH"])

        # Ensure INCLUDE_FINS is passed to the script
        if "INCLUDE_FINS" not in self.params:
            self.params["INCLUDE_FINS"] = True

        if "MULTI_LOBE_OFFSET_FACTOR" in self.params:
            del self.params["MULTI_LOBE_OFFSET_FACTOR"]

        for key, value in self.params.items():
            if isinstance(value, str):
                if key in ["OUTPUT_DIRECTORY", "FINAL_OBJECT_NAME"]:
                    safe_value = value.replace(os.path.sep, '/')
                    script_content.append(f"{key} = r'{safe_value}'")
                else:
                    script_content.append(f"{key} = '{value}'")
            elif isinstance(value, float):
                script_content.append(f"{key} = {int(value) if value.is_integer() else value}")
            elif isinstance(value, bool):
                script_content.append(f"{key} = {value}")
            else:
                script_content.append(f"{key} = {value}")

        script_content.append(f"DIRECTORY_PATH = r'{DIR_PATH}'")
        script_content.append(f"OUTPUT_FORMAT = r'{export_format}'")
        script_content.append(f"OUTPUT_FILE = r'{os.path.abspath(os.path.join(safe_output_dir_host, export_file)).replace(os.path.sep, '/')}'")

        script_content.append("# INPUT PARAMETERS END\n")
        script_content.append(BASE_SCRIPT)
        script_content.append("print('Finished script. Waiting for user to exit Salome GUI.')" if open_gui else "print('Finished TUI execution. Exiting Salome.')")

        with open(script_path, 'w') as f:
            f.write('\n'.join(script_content))

        return script_path

    def run_salome(self, open_gui, export_file, export_format, export_args=None):
        script_path = self._generate_salome_script(export_file, export_format, open_gui)
        salome_launcher = os.path.normpath(self.salome_path)

        if not os.path.exists(salome_launcher):
            raise FileNotFoundError(f"Salome launcher not found at: {salome_launcher}. Please check the path configured in airship_gui.py.")

        mode_flag = "-g" if open_gui else "-t"
        script_path_safe = os.path.normpath(script_path).replace(os.path.sep, '/')

        salome_command = f'"{salome_launcher}" {mode_flag} python {script_path_safe}'
        command_str = f'cmd /C "{salome_command}"'

        print(f"Executing Salome command (via CMD /C): {command_str}")
        print("---")

        try:
            process = subprocess.run(
                command_str,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                shell=True
            )

            if process.stderr:
                print("\n!!! ERROR CAPTURED BY PYTHON PROCESS !!!")
                print(process.stderr)
                print("!!! END ERROR MESSAGE !!!\n")

            if process.stdout:
                print("\n--- Console Output (STDOUT) ---")
                print(process.stdout)
                print("----------------------------\n")

            return process

        except FileNotFoundError:
            raise FileNotFoundError(f"Salome launcher not found at: {salome_launcher}. Check path in airship_gui.py.")
        except Exception as e:
            raise RuntimeError(f"Salome execution failed unexpectedly. Error: {e}")
