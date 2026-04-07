import os
import subprocess
import math
import numpy as np

from geometry_handler import GertlerEnvelope
from meshlab_handler import apply_filters, get_meshdata
from added_mass import compute_added_mass
from airfoil import get_airfoil_points

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not found. Plotting functions will be mocked.")

# --- CONSTANTS ---

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_SCRIPT_FILE = "salome_script.py"
BREAKER = "# INPUT PARAMETERS END"
# Load the base Salome script template
with open(os.path.join(DIR_PATH, BASE_SCRIPT_FILE), 'r') as f:
    BASE_SCRIPT = f.read().split(BREAKER, 1)[1]

class AirshipGeometry:

    def __init__(self, params, salome_path):
        self.params = params
        self.salome_path = salome_path
        self.output_directory = os.path.normpath(params["OUTPUT_DIRECTORY"])
        self.L = params["ENVELOPE_LENGTH"]
        self.l2d = params["l2d"]
        self.D_MAX = self.L / self.l2d
        self.R_MAX = self.D_MAX / 2.0

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def _generate_salome_script(self, export_format):
        """Generates a temporary Salome script with injected user parameters."""
        script_filename = self.params["FINAL_OBJECT_NAME"] + "_salome_script.py"
        script_path = os.path.join(self.output_directory, script_filename)

        script_content = ["# INPUT PARAMETERS START"]

        # Handle Volumetric Dimensioning (for Revolved Profiles)
        if "VOLUME" in self.params and self.params.get("ENVELOPE_SERIES") != "DRAGON_DREAM":
            self.params["ENVELOPE_LENGTH"] = GertlerEnvelope.from_parameters_volume(
                self.params["ENVELOPE_PARAMS"],
                self.params["VOLUME"],
                self.params["ENVELOPE_RESOLUTION"],
                self.params["LOBE_NUMBER"],
                self.params["LOBE_OFFSET_X"],
                self.params["LOBE_OFFSET_Y"],
                self.params["LOBE_OFFSET_Z"]
            ).length

        # Set default values for Salome script injection
        self.params.setdefault("ENVELOPE_TRUNCATION_RATIO", 0)

        # Use .get() inside .setdefault() to ensure we don't trigger KeyError if ENVELOPE_PARAMS isn't available
        self.params.setdefault("CENTRAL_LOBE_PARAMS", self.params.get("ENVELOPE_PARAMS", (0.5, 0.5, 0.5, 0.66, 1.0)))
        self.params.setdefault("CENTRAL_LOBE_LENGTH", self.params.get("ENVELOPE_LENGTH", 100))
        self.params.setdefault("INCLUDE_FINS", True)
        self.params.setdefault("INCLUDE_WINGS", False)

        # --- DRAGON DREAM SAFEGUARDS ---
        self.params.setdefault("HULL_WIDTH", 29.5)
        self.params.setdefault("HULL_HEIGHT", 14.0)
        self.params.setdefault("BOTTOM_FLATNESS", 0.25)
        # -------------------------------

        for key, value in self.params.items():
            if isinstance(value, str):
                script_content.append(f"{key} = r'{value}'")
            elif isinstance(value, float):
                script_content.append(f"{key} = {int(value) if value.is_integer() else value}")
            elif isinstance(value, bool):
                script_content.append(f"{key} = {value}")
            else:
                script_content.append(f"{key} = {value}")

        script_content.append(f"DIRECTORY_PATH = r'{DIR_PATH}'")
        # Ensure Salome knows which format to export
        actual_format = "STL" if export_format == "FULL" else export_format
        script_content.append(f"OUTPUT_FORMAT = r'{actual_format}'")

        script_content.append("# INPUT PARAMETERS END\n")
        script_content.append(BASE_SCRIPT)

        with open(script_path, 'w') as f:
            f.write('\n'.join(script_content))

        return script_path

    def run_salome(self, export_format, open_gui=False):
        """Executes the Salome script and processes the resulting mesh."""
        script_path = self._generate_salome_script(export_format)
        salome_launcher_path = os.path.normpath(self.salome_path)

        if not os.path.exists(salome_launcher_path):
            raise FileNotFoundError(f"Salome launcher not found at: {salome_launcher_path}.")
        
        mode_flag = "-g" if open_gui else "-t"
        script_path_safe = os.path.normpath(script_path).replace(os.path.sep, '/')

        salome_command = f'"{salome_launcher_path}" {mode_flag} python {script_path_safe}'
        command_str = f'cmd /C "{salome_command}"'

        print("[STATUS] Launching Salome subprocess...")

        try:
            process = subprocess.run(
                command_str,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                shell=True
            )

            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print(f"[SALOME ERROR] {process.stderr}")

            final_obj_name = self.params["FINAL_OBJECT_NAME"]
            output_file_lobes = os.path.join(self.output_directory, f"{final_obj_name}_lobes.stl")

            # Setup MeshLab decimation filters
            filters = {}
            if "MESH_TARGET_FACES" in self.params:
                filters["targetfacenum"] = int(self.params["MESH_TARGET_FACES"])
            else:
                filters["targetfacenum"] = 50

            # --- Added Mass Calculation Logic ---
            added_mass_matrix = None

            # BYPASS LOGIC: Only compute if export_format is specifically "FULL"
            if export_format == "FULL":
                if os.path.exists(output_file_lobes):
                    print("[STATUS] Loading lobe mesh for Added Mass calculation...")
                    meshdata = get_meshdata(output_file_lobes, **filters)
                    print("[STATUS] Computing Added Mass Matrix (Vectorized)...")
                    added_mass_matrix = compute_added_mass(*meshdata) #
                else:
                    print("[WARNING] Output file for lobes export not found. Skipping Added Mass.")
            else:
                print("[STATUS] Added Mass calculation skipped by user.")

            return process, added_mass_matrix

        except Exception as e:
            raise RuntimeError(f"Salome execution failed: {e}")

    def geometric_properties(self):
        """Calculates theoretical geometric values including hull, fins, and wings."""
        envelope = GertlerEnvelope.from_parameters(
            self.params["ENVELOPE_PARAMS"],
            self.params["ENVELOPE_LENGTH"],
            self.params["ENVELOPE_RESOLUTION"]
        )

        lobe_number = self.params["LOBE_NUMBER"]
        e = self.params["LOBE_OFFSET_X"]
        f = self.params["LOBE_OFFSET_Y"]
        g = self.params["LOBE_OFFSET_Z"]

        # 1. Calculate Hull Base Properties
        if lobe_number == 1:
            vol, surf, top, side, cv = (envelope.volume(), envelope.surface_area(),
                                        envelope.side_projected_area(), envelope.side_projected_area(),
                                        envelope.cv())
        elif lobe_number == 2:
            vol, surf, top, side, cv = (envelope.volume_bilobe(f), envelope.surface_area_bilobe(f),
                                        envelope.top_projected_area_bilobe(f), envelope.side_projected_area(),
                                        envelope.cv_bilobe(f))
        else:
            vol, surf, top, side, cv = (envelope.volume_trilobe(e, f, g), envelope.surface_area_trilobe(e, f, g),
                                        envelope.top_projected_area_trilobe(e, f, g), envelope.side_projected_area_trilobe(e, f, g),
                                        envelope.cv_trilobe(e, f, g))

        # 2. Add Fin Contributions (if enabled)
        if self.params.get("INCLUDE_FINS", True):
            n_fins = self.params.get("FIN_NUMBER", 4)
            rc = self.params.get("FIN_RC_LENGTH", 0)
            h = self.params.get("FIN_HEIGHT", 0)
            taper = self.params.get("FIN_TAPER_RATIO", 1)
            thick_ratio = self.params.get("FIN_THICKNESS", 0) / 100.0

            fin_planform_area = 0.5 * (rc + rc * taper) * h
            surf += (2 * fin_planform_area * n_fins)
            fin_vol = fin_planform_area * (rc * thick_ratio)
            vol += (fin_vol * n_fins)
            side += (fin_planform_area * 2)
            top += (fin_planform_area * 2)

        # 3. Add Wing Contributions (if enabled)
        if self.params.get("INCLUDE_WINGS", False):
            span = self.params.get("WING_SPAN", 20.0)
            cr = self.params.get("WING_ROOT_CHORD", 5.0)
            ct = self.params.get("WING_TIP_CHORD", 2.0)
            thick_ratio = self.params.get("WING_THICKNESS", 12.0) / 100.0
            dihedral_deg = self.params.get("WING_DIHEDRAL", 5.0)

            # Calculate true physical span accounting for dihedral stretch
            import math
            dihedral_rad = math.radians(dihedral_deg)
            true_span = span / math.cos(dihedral_rad) if math.cos(dihedral_rad) != 0 else span

            total_wing_planform_area = 0.5 * (cr + ct) * true_span
            surf += (2 * total_wing_planform_area)

            avg_chord = (cr + ct) / 2
            wing_vol = total_wing_planform_area * (avg_chord * thick_ratio)
            vol += wing_vol

            top += 0.5 * (cr + ct) * span
            side += (cr * thick_ratio * span * 0.1) + (true_span * math.sin(dihedral_rad) * avg_chord * 0.5)

        return vol, surf, top, side, cv


def plot_and_save_profile(params, length, nx, num_petals, nc, filename, shape_name="Airship_Geometry"):
    """Generates the DAT file and profile plot for the developed gore/petal."""
    coords_2D = GertlerEnvelope.from_parameters(params, length, nx).petal_coordinates(num_petals, nc)

    try:
        with open(filename, 'w') as f:
            f.write(f"# 2D Developed Petal Coordinates\n")
            f.write(f"# X_coordinate   C_coordinate\n")
            for x, c in coords_2D:
                f.write(f"{x:.6f}\t{c:.6f}\n")
        dat_result = f"Coordinates saved to DAT file."
    except Exception as e:
        dat_result = f"Error saving DAT: {e}"

    plot_result = ""
    if plt is None:
        plot_result = "Matplotlib not found."
    else:
        try:
            X_flat = [c[0] for c in coords_2D]
            C_flat = [c[1] for c in coords_2D]
            X_grid = np.array(X_flat).reshape((nx, nc))
            C_grid = np.array(C_flat).reshape((nx, nc))

            png_filename = os.path.splitext(filename)[0] + "_petal.png"

            plt.figure(figsize=(10, 8))
            plt.plot(X_grid[:, 0], C_grid[:, 0], 'r-', label='Longitudinal Edge', linewidth=2)
            plt.plot(X_grid[:, -1], C_grid[:, -1], 'r-', linewidth=2)
            plt.plot(X_grid[:, nc // 2], C_grid[:, nc // 2], 'k--', label='Petal Centerline', linewidth=1)
            plt.title(f'2D Developed Petal: {shape_name}', fontsize=16)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.axis('equal')
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()
            plot_result = f"Plot saved as PNG."
        except Exception as e:
            plot_result = f"Error generating plot: {e}"

    return f"{dat_result}\n{plot_result}"
