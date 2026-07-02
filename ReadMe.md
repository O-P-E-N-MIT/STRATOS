# STRATOS
**S**oftware-based **T**ool for **R**apid **A**irship **T**ranslation & **O**ptimized **S**haping

STRATOS is a comprehensive, open-source Python framework featuring a graphical user interface designed for the rapid parametric modeling, aerodynamic evaluation, and structural analysis of Lighter-Than-Air (LTA) platforms. 

## 🚀 Key Features

### 1. Parametric Envelope Geometry
* **Profile Series Support:** Generate high-fidelity hull shapes using standard series including Gertler, NACA (symmetric), and the Dragon Dream profile.
* **Multi-Lobe Configurations:** Native support for monolobe, bilobe, and trilobe configurations, with customizable lateral and vertical offsets for complex hull combinations.
* **Volumetric Dimensioning:** Automatically scale and dimension envelope lengths to meet specific target volumes and payload requirements.
* **Geometric Properties:** Instantly calculates surface area, volume, projected areas (top/side), and Center of Volume (CV) for any hull configuration.

### 2. Superpressure Balloon Modeling
* **Base Shapes:** Generate Spherical, Prolate, and Oblate balloon geometries based on target volume and aspect ratio.
* **Advanced Gore Models:** Apply distinct gore modeling techniques including Pumpkin, Flat Facet, and Smooth Bumpy parameterizations.
* **Bulge Amplitude & Power:** Fine-tune balloon aesthetics and structural boundaries through customizable bulge amplitudes and fade powers.

### 3. Appendage Design (Wings & Fins)
* **Airfoil Asset Manager:** Import custom `.dat` files or generate NACA 4-digit profiles. Uses advanced fitting methods (CST and PARSEC) to parameterize imported coordinates.
* **Parametric Fins:** Automatically loft stabilizing fins with customizable root chord, height, taper ratio, sweep angle, and angular placement around the hull.
* **Complex Wing Lofting:** Generate full wing geometries directly on the hull using multi-station inputs (span, chord, sweep, dihedral, and twist). 

### 4. Aerostat Performance Analysis
* **Atmospheric Modeling:** Integrated ISA atmospheric calculations for accurate lift estimations based on operational height, relative humidity, and lifting gas properties (purity, delta P, delta T).
* **Ballonet Sizing:** Configure the number of ballonets and calculate required ballonet volume and fabric mass for target operational altitudes.
* **Thermal & Stress Analysis:** Estimates envelope temperature and thermal stress based on solar flux, wind speed, material emissivity, absorptivity, and base strength.
* **Lifespan Tracking:** Projects material degradation using fatigue and UV degradation factors to estimate burst altitude and safety factors.

### 5. Hydrodynamic Added Mass Computation
* **BEM Integration:** Features a built-in potential flow solver using the Boundary Element Method (BEM).
* **6x6 Matrix Output:** Computes the full 6x6 added mass matrix directly from the generated 3D surface mesh, essential for dynamic stability and control simulations.

### 6. Manufacturing & CAD Export
* **Salome Integration:** Operates Salome in the background for robust Boolean operations, surface lofting, and precise CAD solid generation.
* **2D Petal Generation:** Flattens 3D hull geometries into 2D developed petal coordinates (distinct from the volumetric lobes) exported as `.dat` and `.png` files for physical envelope manufacturing.
* **Multi-Format Export:** Directly outputs industry-standard `.STL`, `.STEP`, and `.BREP` files for immediate use in CFD, FEA, or manufacturing pipelines.
* **Real-time 3D Preview:** Integrated PyVista plotting for interactive 3D visualization of the hull, edges, and appendages within the GUI.

## 💻 OS Compatibility
STRATOS relies heavily on the Salome CAD engine for its solid modeling backend.
* **Windows:** Fully supported natively.
* **Linux:** Highly feasible. Requires minor adjustments to the file dialog configurations and subprocess commands to point to Linux shell binaries.
* **MacOS:** Challenging. Salome does not offer native MacOS binaries. Mac users will currently need to run the CAD backend through a Linux Virtual Machine or Docker container.

## 🛠️ System Requirements
* Python 3.x
* **Salome:** Required for Boolean operations and STEP/BREP generation. 

### Python Libraries Required
* **PySide6**: Framework for the graphical user interface (GUI).
* **PyVista & pyvistaqt**: For 3D visualization and interactive plotting within the interface.
* **PyMeshLab**: For mesh decimation, filtering, and handling 3D surface geometries.
* **NumPy & SciPy**: For core mathematical computations, matrix operations, and potential flow solving.
* **Matplotlib**: For generating 2D performance graphs and petal plots.
* **Shapely**: For complex geometric calculations, specifically trilobe volume estimations.

To install all the required libraries, run the following command in your terminal or command prompt:
```bash
pip install PySide6 pyvista pyvistaqt pymeshlab numpy scipy matplotlib 
```
## 📥 Installation & Setup
1. Clone the repository to your local machine.
2. Install the required Python dependencies using the `pip` command provided above.
3. **Download the CAD Engine:** STRATOS requires Salome to process solid 3D geometry. Download the Windows version from the Official Salome Website.
4. Extract the downloaded Salome archive to a memorable location on your drive.
5. Run `airship_gui.py` to launch the STRATOS application.
6. Upon first launch, the GUI will prompt you to locate your `run_SALOME.bat` executable (found inside your extracted Salome folder) to link the CAD backend.

## 💡 Usage Workflow
STRATOS is driven by an intuitive tabbed interface. A standard workflow looks like this:

1. **Select Dimensioning Mode:** At the top of the interface, choose between Standard Mode (Length-based), Volumetric Mode, Super Pressure Balloon, or Aerostat.
2. **Configure Hull Geometry:** In the **Envelope Geometry** tab, pick a profile series (Gertler, NACA, Dragon Dream) and adjust parameters like L/D, prismatic coefficient, and nose/tail radii.
3. **Design Appendages:** Navigate to the **Airfoil Design**, **Wing Design**, or **Fin Design** tabs to import custom airfoils, loft wings, or place stabilizing fins onto your hull.
4. **Aerostat Analysis (Optional):** If in Aerostat mode, input environmental conditions and lifting gas properties to generate performance graphs and calculate burst altitude.
5. **Generate & Export:** Go to the **Output** tab, set your project name, choose your export format (STL, STEP, BREP), and click **RUN GENERATION**. View the 3D solid result directly in the built-in PyVista viewer, alongside the computed 6x6 added mass matrix!

## 📑 Cite As
> Pedapudi Anantha Hari Arun, Sudarsan D. Naidu, Pranav Mittal, and Manikandan Murugaiah "STRATOS: Software-based Tool for Rapid Airship Translation & Optimized Shaping", [https://github.com/O-P-E-N-MIT/STRATOS](https://github.com/O-P-E-N-MIT/STRATOS)

## 📄 License
This project is distributed under the MIT License. See the `LICENSE` file for more information.
