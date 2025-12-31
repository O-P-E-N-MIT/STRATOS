import os
import subprocess
import math
import numpy as np

from geometry_handler import GertlerEnvelope

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not found. Plotting functions will be mocked.")

# --- CONSTANTS ---

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_SCRIPT_FILE = "salome_script.py"
BREAKER = "# INPUT PARAMETERS END"
BASE_SCRIPT = open(os.path.join(DIR_PATH, BASE_SCRIPT_FILE), 'r').read().split(BREAKER, 1)[1]

class AirshipGeometry:

    def __init__(self, params, salome_path):
        self.params = params
        self.salome_path = salome_path
        self.output_directory = params["OUTPUT_DIRECTORY"]
        self.L = params["ENVELOPE_LENGTH"]
        self.l2d = params["l2d"]
        self.D_MAX = self.L / self.l2d
        self.R_MAX = self.D_MAX / 2.0
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

        with open(script_path, 'w') as f:
            f.write('\n'.join(script_content))

        return script_path

    def run_salome(self, open_gui, export_file, export_format):
        script_path = self._generate_salome_script(export_file, export_format, open_gui)
        salome_launcher = os.path.normpath(self.salome_path)

        if not os.path.exists(salome_launcher):
            raise FileNotFoundError(f"Salome launcher not found at: {salome_launcher}. Please check the path configured in airship_gui.py.")

        mode_flag = "-g" if open_gui else "-t"
        script_path_safe = os.path.normpath(script_path).replace(os.path.sep, '/')

        salome_command = f'"{salome_launcher}" {mode_flag} python {script_path_safe}'
        command_str = f'cmd /C "{salome_command}"'

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
    
    # Returns the geometric properties of a selected lobe.
    # (volume, surface area, top projected area, side projected area)
    def geometric_properties (self):
        envelope = GertlerEnvelope.from_parameters(self.params["ENVELOPE_PARAMS"], self.params["ENVELOPE_LENGTH"], self.params["ENVELOPE_RESOLUTION"])
        
        lobe_number = self.params["LOBE_NUMBER"]
        e = self.params["LOBE_OFFSET_X"]
        f = self.params["LOBE_OFFSET_Y"]
        g = self.params["LOBE_OFFSET_Z"]

        if lobe_number == 1:
            return envelope.volume(), envelope.surface_area(), envelope.side_projected_area(), envelope.side_projected_area()
        elif lobe_number == 2:
            return envelope.volume_bilobe(f), envelope.surface_area_bilobe(f), envelope.top_projected_area_bilobe(f), envelope.side_projected_area()
        else:
            return envelope.volume_trilobe(e, f, g), envelope.surface_area_trilobe(e, f, g), envelope.top_projected_area_trilobe(e, f, g), envelope.side_projected_area_trilobe(e, f, g)

# A function to plot and save petal coordinates.
def plot_and_save_profile (params, length, nx, num_petals, nc, filename, shape_name="Airship_Geometry"):
    coords_2D = GertlerEnvelope.from_parameters(params, length, nx).petal_coordinates(num_petals, nc)

    try:
        with open(filename, 'w') as f:
            f.write(f"# 2D Developed Petal Coordinates\n")
            f.write(f"# X-Points: {nx}, C-Points: {nc}\n")
            f.write(f"# X_coordinate   C_coordinate\n")

            for x, c in coords_2D:
                f.write(f"{x:.6f}\t{c:.6f}\n")

        dat_result = f"Coordinates successfully saved to DAT file:\n{filename}"

    except Exception as e:
        dat_result = f"Error generating DAT file:\n{e}"
    
    plot_result = ""

    if plt is None:
        plot_result = "\nMatplotlib not available. Cannot generate 2D plot."
    else:
        try:
            X_flat = [c[0] for c in coords_2D]
            C_flat = [c[1] for c in coords_2D]
            X_grid = np.array(X_flat).reshape((nx, nc))
            C_grid = np.array(C_flat).reshape((nx, nc))

            png_filename = os.path.splitext(filename)[0] + "_petal.png"

            plt.figure(figsize=(10, 8))

            plt.plot(X_grid[:, 0], C_grid[:, 0], 'r-', label='Longitudinal Edge (Boundary)', linewidth=2)
            plt.plot(X_grid[:, -1], C_grid[:, -1], 'r-', linewidth=2)

            plt.plot(X_grid[:, nc // 2], C_grid[:, nc // 2], 'k--', label='Petal Centerline', linewidth=1)

            plt.plot(X_grid[0, :], C_grid[0, :], 'b-', label='Axial Boundary', linewidth=2)
            plt.plot(X_grid[nx-1, :], C_grid[nx-1, :], 'b-', linewidth=2)

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