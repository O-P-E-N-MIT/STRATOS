import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

from plotter import GertlerEnvelope, STANDARD_ENVELOPES

# Main directory path where the executing script file is located
# NOTE: The base script file and this file must be in the same directory for this.
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# Base salome script file
BASE_SCRIPT_FILE = "salome_script.py"

# The temporary script file to be executed in Salome
# TODO: Allow users to choose a script file of custom name (if necessary)
EXEC_SCRIPT_FILE = "exec_salome_temp.py"

# The line in the base script file which indicates the end of input parameters
BREAKER = "# INPUT PARAMETERS END"

# Extract the base script content excluding the input parameters
BASE_SCRIPT = open(os.path.join(DIR_PATH, BASE_SCRIPT_FILE), 'r').read().split(BREAKER, 1)[1]

# List of parameters to be checked.
# NOTE: The value corresponding to the parameter is the default one while "None" means that the parameter is a required one.
PARAMETER_CHECK = {
    "ENVELOPE_LENGTH": None,
    "ENVELOPE_RESOLUTION": 100,
    "ENVELOPE_TRUNCATION_RATIO": 0,

    "LOBE_NUMBER": 1,
    "LOBE_OFFSET_X": 0,
    "LOBE_OFFSET_Y": 0,
    "LOBE_OFFSET_Z": 0,

    "FIN_AXIAL_OFFSET": None,
    "FIN_THICKNESS": None,
    "FIN_RC_LENGTH": None,
    "FIN_SECTION_RESOLUTION": 50,
    "FIN_TAPER_RATIO": 1,
    "FIN_HEIGHT": None,
    "FIN_SWEEP_ANGLE": 0,
    "FIN_TIP_ANGLE": 0,
    "FIN_NUMBER": 4,

    "SHEET_LENGTH_RATIO": 0
}

class AirshipGeometry:

    # List of all parameters
    #
    # ENVELOPE_PARAMS               Gertler parameters of the envelope
    # ENVELOPE_LENGTH               Length of the envelope
    # ENVELOPE_DIAMETER             Diameter of the envelope (either this or length has to be provided)
    # ENVELOPE_RESOLUTION           Number of points to be taken to describe the envelope. (optional, default = 100)
    # ENVELOPE_TRUNCATION_RATIO     Ratio of length to the total envelope length to be truncated. (optional, default = 0)
    #
    # LOBE_NUMBER                   Number of lobes (optional, default = 1)
    # LOBE_OFFSET_X                 Distance of central lobe from YZ plane (optional)
    # LOBE_OFFSET_Y                 Distance of extreme lobe from XZ plane (optional)
    # LOBE_OFFSET_Z                 Distance of central lobe from XY plane (optional)
    #
    # FIN_RC_LENGTH                 Root chord length of the fin
    # FIN_AXIAL_OFFSET              Ratio of the distance between the nose of the extreme lobe and the leading edge of the root chord of the fin to the extreme lobe length.
    # FIN_THICKNESS                 Ratio of the thickness of the symmetric NACA airfoil to it's chord length
    # FIN_HEIGHT                    Height of the fin
    # FIN_NUMBER                    Number of fins (optional, default = 4, must be even in case of multi lobe)
    # FIN_TAPER_RATIO               Ratio of tip chord of the fin to its root chord (optional, default = 1)
    # FIN_SWEEP_ANGLE               Angle between the midchord and the normal to the hull at that point in degrees (optional, default = 0)
    # FIN_TIP ANGLE                 Angle between the tip airfoil with respect to root airfoil in degrees (optional, default = 0)
    # FIN_THETA_POS                 Angle positions of each fin with respect to +Z in degrees (optional, default is evenly distributed).
    #                               In case of multi lobe design, positions for only half of the fins on one side of the extreme lobe is to be provided.
    # FIN_SECTION_RESOLUTION        Number of points used to represent the airfoil (optional, default = 50)
    # 
    # SHEET_LENGTH_RATIO            Ratio of the length of the extreme lobe length to be covered by the fairing sheets.
    # FINAL_OBJECT_NAME             Name of the salome object (optional, default = Airship)
    #
    # The following central lobe parameters are for trilobe design in case of central lobe having different geometries than the extreme lobe.
    # 
    # CENTRAL_LOBE_PARAMS           Gertler parameters of the central lobe (optional)
    # CENTRAL_LOBE_LENGTH           Length of the central lobe (optional)
    # CENTRAL_LOBE_DIAMETER         Diameter of the central lobe (optional)
    def __init__ (self, parameters, salome_exec_path):
        self.parameters = parameters
        self.salome_exec_path = salome_exec_path

        # Check for all the basic parameters and assign if the default values if not required.
        for key, value in PARAMETER_CHECK.items():
            if key not in parameters:
                if value is None:
                    raise Exception(f"AirshipGeometry: {key} not specified.")
                else:
                    parameters[key] = value

        lobe_number = self.parameters["LOBE_NUMBER"]
        fin_number = self.parameters["FIN_NUMBER"]

        # To maintain symmetry, there must be only even number of fins in multi lobe design.
        if lobe_number > 1 and fin_number % 2 != 0:
            raise Exception("AirshipGeometry: There must be even fin number for multi lobe geometry.")

        # If fin theta positions are not specified, they are evenly distributed.
        if "FIN_THETA_POS" not in parameters:
            parameters["FIN_THETA_POS"] = [i * 360 / fin_number for i in range(0, fin_number)] if lobe_number == 1 else [i * 360 / (fin_number - 2) for i in range(0, int(fin_number/2))]

        # Initiate the extreme lobe.
        self.envelope = self.init_lobe("ENVELOPE")

        # Initiate the central lobe if it is a trilobe design.
        if lobe_number == 3:
            self.central_lobe = self.init_lobe("CENTRAL_LOBE", True)
        
        # In case of running in GUI, if the users wants a custom name for the Salome object.
        if "FINAL_OBJECT_NAME" not in self.parameters:
            self.parameters["FINAL_OBJECT_NAME"] = "Airship"
        
        # Set the directory path.
        self.parameters["DIRECTORY_PATH"] = DIR_PATH
    
    # TODO: This entire thing is overcomplicated and can be simplified I believe
    def init_lobe (self, id, is_central = False):
        res = self.parameters["ENVELOPE_RESOLUTION"]
        len_key = f"{id}_LENGTH"
        dia_key = f"{id}_DIAMETER"
        envelope = None

        # If neither length or diameter is provided and is a central lobe, keep the dimensions same as the extreme lobe.
        if is_central and (len_key not in self.parameters) and (dia_key not in self.parameters):
            self.parameters[len_key] = self.envelope.length

        # Check if the Gertler parameters are given for the lobe geometry.
        if (params := self.parameters.get(f"{id}_PARAMS")) is not None:
            l2d = params[4]
            len = self.parameters[len_key] if len_key in self.parameters else self.parameters[dia_key] * l2d
            envelope = GertlerEnvelope.from_parameters(params, len, res)

        # If it is a central lobe and the geometry is not explicitly provided.
        elif is_central:
            l2d = self.envelope.length / self.envelope.diameter
            len = self.parameters[len_key] if len_key in self.parameters else self.parameters[dia_key] * l2d
            envelope = GertlerEnvelope(self.envelope.coeffs, len, len / l2d, res)

        else:
            raise Exception(f"AirshipGeometry: No proper shape parameters have been specified for {id}")
        
        # Assign all the required parameters for generation of the lobe.
        self.parameters[f"{id}_COEFFS"] = envelope.coeffs
        self.parameters[len_key] = envelope.length
        self.parameters[dia_key] = envelope.diameter

        return envelope
    
    # Generates the salome script to be executed.
    def script (self):
        param_lines = ""

        for key, value in self.parameters.items():
            if isinstance(value, str):
                param_lines += f'{key} = "{value}"\n'
            elif isinstance(value, list) or isinstance(value, tuple):
                param_lines += f'{key} = [{", ".join([str(x) for x in value])}]\n'
            else:
                param_lines += f'{key} = {value}\n'
        
        return f"{param_lines}\n\n{BASE_SCRIPT}\n"
    
    # Creates a temporary salome script file and executes it.
    #
    # NOTE: Allowed export formats are STEP, IGES, BREP, STL, XAO. For STL format, absolute and relative deflection arguments
    # are to be specified via export_args.
    # TODO: Besides STL, check whatever extra export_arguments are required for other formats.
    # TODO: Most of the testing is done only in Windows. Before publishing it, it is better that we test this in various
    # other platforms and check what adjustments have to be made.
    def run_salome (self, open_gui=False, remove_temp_script=True, export_file=None, export_format=None, export_args=()):
        exec_script_path = os.path.join(DIR_PATH, EXEC_SCRIPT_FILE)
        exec_script = self.script()

        # Add the code to export the geom at the end of the file.
        if export_format:
            export_file = os.path.join(DIR_PATH, export_file or f"{self.parameters['FINAL_OBJECT_NAME']}_export.{export_format.lower()}")
            exec_script += f"geompy.Export{export_format.upper()}(airship, '{export_file}', {",".join(str(arg) for arg in export_args)})"

        open(exec_script_path, 'w').write(exec_script)
        
        subprocess.run([self.salome_exec_path, "-g" if open_gui else "-t", "python", exec_script_path], check=True)
        
        # Removing the temporary file.
        if remove_temp_script:
            os.remove(exec_script_path)

# A function to plot the petal profile and save it.
def plot_petal_profile (envelope, num_petals, num_points_c, dat_filename, shape_name='Envelope'):
    coords_2D = envelope.petal_coordinates(num_petals, num_points_c)
    num_points_x = envelope.n

    """Plots the 2D Developed Petal (X vs Circumferential Distance C) and saves it as a PNG."""
    try:
        # Reshape the coordinates back into a grid for plotting the boundaries
        X_flat = [c[0] for c in coords_2D]
        C_flat = [c[1] for c in coords_2D]
        X_grid = np.array(X_flat).reshape((num_points_x, num_points_c))
        C_grid = np.array(C_flat).reshape((num_points_x, num_points_c))

        png_filename = os.path.splitext(dat_filename)[0] + ".png"

        plt.figure(figsize=(10, 8))

        # Plot the side edges (longitudinal lines) - these define the petal shape
        plt.plot(X_grid[:, 0], C_grid[:, 0], 'r-', label='Longitudinal Edge (Boundary)', linewidth=2)
        plt.plot(X_grid[:, -1], C_grid[:, -1], 'r-', linewidth=2)

        # Plot the centerline (where C=0)
        plt.plot(X_grid[:, num_points_c // 2], C_grid[:, num_points_c // 2], 'k--', label='Petal Centerline', linewidth=1)

        # Plot the axial lines (cross-sections) for start and end
        # Since R(0)=0 and R(L)=0, these lines now correctly collapse to points at the origin/terminus.
        plt.plot(X_grid[0, :], C_grid[0, :], 'b-', label='Axial Boundary', linewidth=2)
        plt.plot(X_grid[num_points_x-1, :], C_grid[num_points_x-1, :], 'b-', linewidth=2)

        plt.title(f'2D Developed Petal: {shape_name} ({num_petals} Petals)', fontsize=16)
        plt.xlabel('Axial Position (X) [units]', fontsize=12)
        plt.ylabel('Circumferential Distance (C) [units]', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.axis('equal') # Crucial for showing the correct tapered shape

        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.close()

        return f"\nPlot successfully saved to:\n{png_filename}"

    except Exception as e:
        return f"\nError generating plot:\n{e}"