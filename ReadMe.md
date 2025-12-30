## Framework for Airship Geometry Generator
<a name="top"></a>

[![OS](https://img.shields.io/badge/OS-linux%2C%20windows%2C%20macOS-0078D4)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

:star: Star us on GitHub - it is what keeps us going!

### About

The Airship Geometry Generator is an open-source Python-based framework designed to facilitate the design and 3D modeling of advanced airship hull geometries. It specializes in Gertler Envelope parametrization, allowing users to define complex aerodynamic shapes through specific geometric coefficients, such as maximum thickness position, nose/tail radii, and prismatic coefficients. The framework integrates with SALOME for automated 3D CAD generation and features a developed 2D petal (gore) generator for manufacturing preparation.

### Features

* User-friendly and easily accessible GUI.

* Support for Conventional, Bilobe, and Trilobe hull design with adjustable separation offset.

* Precise control over hull shapes using $m1$, $r0$, $r1$, $cp$ and $l/d$ coefficients.

* Automated generation of fins based on NACA 4-digit airfoil profile with adjustable sweep, taper, and axial positioning.

* Automated calculation of hull length based on a target volume requirement and lobe configuration.

* Generates developed 2D petal (gore) coordinates and plots for fabric cutting and manufacturing.

* Scripted interface with SALOME for exporting high-fidelity 3D models in .STL, .BREP, or .STEP formats.

### Pre-requisites

* To use this tool, ensure the following software is installed:

* SALOME: https://www.salome-platform.org/ $\rightarrow$ Verify that the path of the executable is stored in the C:\SALOME-9.15.0.
  
* Python: https://www.python.org/downloads/
  
* Python specific packages $\rightarrow$

    - numpy
    - matplotlib
    - scipy
    - PySide6
    - shapely

### Installation

```bash
python -m pip install -r requirements.txt
```

### Usage

* Open airship_gui.py and update the self.salome_path variable to point to your SALOME installation executable (e.g., run_SALOME.bat).

* Run airship_gui.py
  
```bash
python3 airship_gui.py
```

* Set the Gertler parameters or select a standard preset in the Standard Envelope dropdown.

* Select the number of lobes and set the lobes' separation offsets.

* Configure fin dimensions or uncheck "GENERATE FINS WITH HULL" for a hull-only model.

* Click RUN GENERATION in the Output tab to trigger the SALOME script and export your 3D model.

### Cite As

```bash
Anantha Hari Arun Pedapudi, Sudarsan D. Naidu, and Manikandan Murugaiah "XXXXXX: Framework for Airship Geometry Generator", https://github.com/O-P-E-N-MIT/Airship-Geometric-Modelling
```
### References
1. Manikandan, M., Shah, R. R., Priyan, P., Singh, B., & Pant, R. S. (2023). A parametric design approach for multi-lobed hybrid airships. The Aeronautical Journal, 128(1319), 1–36. https://doi.org/10.1017/aer.2023.37

2. Pai, A., & Manikandan, M. (2025). A comparative study of aerodynamic characteristics of conventional and multi-lobed airships. The Aeronautical Journal, 129(1339), 2435–2459. https://doi.org/10.1017/aer.2025.39

3. Alam, M. I., & Pant, R. S. (2017). Surrogate based shape optimization of airship envelopes. In AIAA | ARC. https://doi.org/10.2514/6.2017-3393
  
4. Ceruti, A., Gambacorta, D., & Marzocca, P. (2018). Unconventional hybrid airships design optimization accounting for added masses. Aerospace Science and Technology, 72, 164–173. https://doi.org/10.1016/j.ast.2017.10.042

