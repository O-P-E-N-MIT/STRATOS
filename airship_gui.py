import os
import re
import sys

# --- 3D VISUALIZATION IMPORTS ---
import pyvista as pv
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSlider, QGroupBox, QFileDialog, QTextEdit,
    QButtonGroup, QCheckBox, QMessageBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from pyvistaqt import BackgroundPlotter

# Integration of balloon.py
from balloon import create_balloon_geometry
from geometry import AirshipGeometry, plot_and_save_profile
from geometry_handler import STANDARD_ENVELOPES

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# --- THREAD-SAFE LOGGING SYSTEM ---

class StatusLogger(QObject):
    """
    Redirects python 'print' statements to a Signal.
    Allows print() calls from external files to show up in the
    GUI QTextEdit without causing cross-thread memory corruption.
    """
    message_logged = Signal(str)

    def write(self, s):
        if s.strip():
            self.message_logged.emit(s.strip())

    def flush(self):
        pass

# --- WORKER OBJECT FOR CALCULATIONS ---

class GenerationWorker(QObject):
    """Handles the heavy generation logic in a background thread."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, mode_id, params, volume_val, gore_model, compute_added_mass=True):
        super().__init__()
        self.mode_id = mode_id
        self.params = params
        self.volume_val = volume_val
        self.gore_model = gore_model
        self.compute_added_mass = compute_added_mass

class GenerationWorker(QObject):
    """Handles the heavy generation logic in a background thread."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, mode_id, params, volume_val, gore_model, compute_added_mass=True):
        super().__init__()
        self.mode_id = mode_id
        self.params = params
        self.volume_val = volume_val
        self.gore_model = gore_model
        self.compute_added_mass = compute_added_mass

    def run(self):
        try:
            # Determine the file path for the primary STL output
            output_path = os.path.join(self.params['OUTPUT_DIRECTORY'], f"{self.params['FINAL_OBJECT_NAME']}.stl")

            if self.mode_id == 3: # BALLOON MODE
                print(f"[PROCESS] Starting Superpressure Balloon generation...")
                # Balloon generation logic remains independent of Salome
                create_balloon_geometry(
                    gore_model=self.gore_model,
                    target_volume=self.volume_val,
                    gores=int(self.params['N_PETALS']),
                    params=self.params['balloon_params'],
                    theta_resolution=int(self.params.get("THETA_RES", 400)),
                    phi_resolution=int(self.params.get("PHI_RES", 600)),
                    output_file=output_path,
                    single_gore=False
                )
                print(f"[SUCCESS] Balloon STL generated at: {output_path}")
                self.finished.emit((None, output_path))

            else: # AIRSHIP MODES
                print(f"[PROCESS] Initializing Airship Geometry Engine...")
                g = AirshipGeometry(self.params, self.params['SALOME_PATH'])

                # LOGIC BYPASS:
                # If compute_added_mass is False, we pass the format (e.g., "STL") to skip BEM
                # If True, we pass "FULL" to trigger compute_added_mass in geometry.py
                if not self.compute_added_mass:
                    fmt = self.params.get('EXPORT_FORMAT', 'STL')
                    process, matrix = g.run_salome(fmt)
                    matrix = None
                else:
                    print("[PROCESS] Computing Added Mass Matrix...")
                    process, matrix = g.run_salome("FULL")

                print(f"[SUCCESS] Airship output generated at: {output_path}")
                self.finished.emit((matrix, output_path))

        except Exception as e:
            self.error.emit(str(e))

# --- UI COMPONENTS ---

class LabeledSlider (QGroupBox):
    value_changed_by_user = Signal(float)
    def __init__(self, label, min_val, max_val, default_val, step, decimals=4, parent=None):
        super().__init__(label, parent)
        self.decimals, self.step = decimals, step
        self.min_val, self.max_val = min_val, max_val
        self._max_slider_val = int((max_val - min_val) / step)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self._max_slider_val)
        self.slider.setValue(int((default_val - min_val) / step))

        self.value_editor = QLineEdit()
        self.value_editor.setFixedWidth(85)
        self.value_editor.setFont(QFont("Monospace", 9))
        self.value_editor.setValidator(QDoubleValidator(min_val, max_val, decimals))

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.slider)
        h_layout.addWidget(self.value_editor)
        layout = QVBoxLayout(self)
        layout.addLayout(h_layout)

        self.slider.valueChanged.connect(self._sync_to_editor)
        self.value_editor.editingFinished.connect(self._sync_to_slider)
        self._sync_to_editor(self.slider.value())

    def _sync_to_editor(self, val):
        float_val = round(self.min_val + val * self.step, self.decimals)
        self.value_editor.setText(f"{float_val:.{self.decimals}f}")
        self.value_changed_by_user.emit(float_val)

    def _sync_to_slider(self):
        try:
            val = float(self.value_editor.text())
            clamped = max(self.min_val, min(self.max_val, val))
            self.slider.setValue(int((clamped - self.min_val) / self.step))
            self.value_changed_by_user.emit(clamped)
        except: pass

    def get_value(self): return float(self.value_editor.text())
    def set_value(self, val):
        slider_pos = int((val - self.min_val) / self.step)
        self.slider.setValue(slider_pos)
        self.value_editor.setText(f"{val:.{self.decimals}f}")

class AirshipGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Airship Geometry Generator | Salome Interface")
        self.setGeometry(100, 100, 1400, 950)

        self.salome_path = r"C:\SALOME-9.15.0\run_SALOME.bat"
        self.base_output_directory = os.path.join(os.path.expanduser("~"), "Documents", "Airship_Outputs")
        if not os.path.exists(self.base_output_directory):
            os.makedirs(self.base_output_directory)

        self.current_session_folder = self.base_output_directory
        self.inputs = {}
        self.setup_style()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self._init_persistent_controls()

        self.tab_widget = QTabWidget()
        self.primary_input_tab = QWidget()
        self.fairings_tab = QWidget()
        self.fin_tab = QWidget()
        self.output_tab = QWidget()

        self.setup_primary_tab_layout()
        self.setup_aerostatics_tab()
        self.setup_fairings_tab()
        self.setup_fin_tab()
        self.setup_output_tab()

        self.main_layout.addWidget(self.tab_widget)
        self.main_layout.addWidget(self.setup_navigation_buttons())

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self.refresh_tabs()
        self.load_defaults()

        # START LOG REDIRECTION
        self.status_logger = StatusLogger()
        self.status_logger.message_logged.connect(self._append_to_log)
        sys.stdout = self.status_logger

    def _on_tab_changed(self, index):
        """Refreshes navigation and attempts to load existing mesh on Output tab."""
        self._update_navigation_buttons()
        if self.tab_widget.tabText(index).startswith("Output"):
            proj_name = self.inputs["FINAL_OBJECT_NAME"].text()
            path = os.path.join(self.current_session_folder, f"{proj_name}.stl")
            self._update_3d_view(path)

    def _append_to_log(self, text):
        self.log.append(text)

    def _init_persistent_controls(self):
        self.mode_button_group = QButtonGroup(self)
        self.mode_btns = []
        for i, name in enumerate(["STANDARD MODE", "VOLUMETRIC MODE", "SUPER PRESSURE BALLOON", "AEROSTATIC"], 1):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            self.mode_button_group.addButton(btn, i)
            self.mode_btns.append(btn)
        self.mode_btns[0].setChecked(True)

        self.lobe_button_group = QButtonGroup(self)
        self.lobe_btns = []
        for i, name in enumerate(["MONOLOBE", "BILOBE", "TRILOBE"], 1):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            self.lobe_button_group.addButton(btn, i)
            self.lobe_btns.append(btn)
        self.lobe_btns[0].setChecked(True)

        for btn in self.mode_btns: btn.clicked.connect(self.refresh_tabs)
        for btn in self.lobe_btns: btn.clicked.connect(self.refresh_tabs)

        self.header_widget = QWidget()
        h_layout = QVBoxLayout(self.header_widget)
        h_layout.setContentsMargins(0, 0, 0, 0)

        m_grp = QGroupBox("Dimensioning Mode")
        m_lay = QHBoxLayout(m_grp)
        for btn in self.mode_btns: m_lay.addWidget(btn)

        self.hull_config_group = QGroupBox("Hull Configuration")
        l_lay = QHBoxLayout(self.hull_config_group)
        for btn in self.lobe_btns: l_lay.addWidget(btn)

        h_layout.addWidget(m_grp)
        h_layout.addWidget(self.hull_config_group)

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #D4D4D4; font-family: Arial; font-size: 10pt; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background: #252526; }
            QTabBar::tab { background: #1e1e1e; padding: 10px 20px; border: 1px solid #3c3c3c; }
            QTabBar::tab:selected { background: #252526; border-top: 3px solid #00BFFF; }
            QGroupBox { border: 1px solid #3c3c3c; margin-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; color: #00BFFF; padding: 0 5px; }
            QLineEdit, QTextEdit, QTableWidget { background-color: #3C3C3C; border: 1px solid #3c3c3c; color: #D4D4D4; }
            QHeaderView::section { background-color: #2D2D2D; color: #00BFFF; padding: 4px; border: 1px solid #3c3c3c; font-weight: bold; }
            QComboBox#LargeDropdown { background-color: #3C3C3C; border: 2px solid #00BFFF; padding: 8px; border-radius: 4px; font-size: 11pt; font-weight: bold; color: #FFFFFF; }
            QPushButton { background-color: #3C3C3C; border: 1px solid #3c3c3c; padding: 8px; border-radius: 4px; color: #D4D4D4; }
            QPushButton:hover { background-color: #505050; border: 1px solid #00BFFF; }
            QPushButton:disabled { background-color: #2a2a2a; color: #555555; border: 1px solid #2a2a2a; }
            QPushButton:checked { background-color: #00BFFF; color: #1e1e1e; font-weight: bold; }
            
            QCheckBox#FinToggle { 
                color: #FFFFFF; 
                font-size: 12pt; 
                font-weight: bold; 
                padding: 10px; 
                background-color: #2D2D2D; 
                border: 2px solid #00BFFF; 
                border-radius: 5px;
            }
            QCheckBox#FinToggle::indicator { width: 25px; height: 25px; }
            QCheckBox#FinToggle:hover { background-color: #3D3D3D; }
        """)

    def setup_primary_tab_layout(self):
        layout = QVBoxLayout(self.primary_input_tab)
        layout.addWidget(self.header_widget)

        self.hull_shape_box = QGroupBox("Hull Envelope Shape")
        cl = QGridLayout(self.hull_shape_box)
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("LargeDropdown")
        self.preset_combo.setMinimumHeight(45)
        self.preset_combo.addItems(list(STANDARD_ENVELOPES.keys()))
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        cl.addWidget(QLabel("Shape Preset:"), 0, 0)
        cl.addWidget(self.preset_combo, 0, 1, 1, 2)

        self.inputs["l2d"] = LabeledSlider("L/D Ratio", 1, 8, 3.266, 0.0001, 4)
        self.inputs["m1"] = LabeledSlider("m1", 0.3, 0.6, 0.419, 0.0001, 4)
        self.inputs["r0"] = LabeledSlider("r0", 0.01, 1, 0.337, 0.0001, 4)
        self.inputs["r1"] = LabeledSlider("r1", 0.01, 1, 0.251, 0.0001, 4)
        self.inputs["cp"] = LabeledSlider("cp", 0.5, 0.8, 0.651, 0.0001, 4)
        self.inputs["ENVELOPE_RESOLUTION"] = LabeledSlider("Resolution", 50, 500, 150, 1, 0)

        cl.addWidget(self.inputs["l2d"], 1, 0)
        cl.addWidget(self.inputs["m1"], 1, 1)
        cl.addWidget(self.inputs["r0"], 2, 0)
        cl.addWidget(self.inputs["r1"], 2, 1)
        cl.addWidget(self.inputs["cp"], 3, 0)
        cl.addWidget(self.inputs["ENVELOPE_RESOLUTION"], 3, 1)
        layout.addWidget(self.hull_shape_box)

        self.balloon_params_box = QGroupBox("Balloon Geometry Parameters")
        bl = QGridLayout(self.balloon_params_box)
        self.inputs["GORE_MODEL"] = QComboBox()
        self.inputs["GORE_MODEL"].setObjectName("LargeDropdown")
        self.inputs["GORE_MODEL"].addItems(["SMOOTH_BUMPY", "PUMPKIN", "OBLATE_LOBED", "FLAT_FACET", "NONE"])

        self.inputs["THETA_RES"] = LabeledSlider("Theta Resolution", 50, 1000, 400, 1, 0)
        self.inputs["PHI_RES"] = LabeledSlider("Phi Resolution", 50, 1000, 600, 1, 0)
        self.inputs["ASPECT_RATIO"] = LabeledSlider("Aspect Ratio", 0.1, 5.0, 1.0, 0.0001, 4)
        self.inputs["BULGE_AMPLITUDE"] = LabeledSlider("Bulge Amplitude", 0, 1.0, 0.0, 0.0001, 4)
        self.inputs["BULGE_POWER"] = LabeledSlider("Bulge Power", 1, 10, 1, 0.0001, 4)
        self.inputs["GORE_AMPLITUDE"] = LabeledSlider("Gore Amplitude", 0, 0.5, 0.05, 0.0001, 4)
        self.inputs["GORE_FADE_POWER"] = LabeledSlider("Gore Fade/Power", 1, 10, 4, 0.0001, 4)

        bl.addWidget(QLabel("Gore Model:"), 0, 0)
        bl.addWidget(self.inputs["GORE_MODEL"], 0, 1)
        bl.addWidget(self.inputs["ASPECT_RATIO"], 1, 0)
        bl.addWidget(self.inputs["GORE_AMPLITUDE"], 1, 1)
        bl.addWidget(self.inputs["GORE_FADE_POWER"], 2, 0)
        bl.addWidget(self.inputs["BULGE_AMPLITUDE"], 2, 1)
        bl.addWidget(self.inputs["BULGE_POWER"], 3, 0)
        bl.addWidget(self.inputs["THETA_RES"], 4, 0)
        bl.addWidget(self.inputs["PHI_RES"], 4, 1)
        layout.addWidget(self.balloon_params_box)

        self.length_box = QGroupBox("Standard Mode: Length")
        ll = QVBoxLayout(self.length_box)
        self.inputs["ENVELOPE_LENGTH"] = LabeledSlider("Length (L)", 10, 500, 100, 0.0001, 4)
        ll.addWidget(self.inputs["ENVELOPE_LENGTH"])
        layout.addWidget(self.length_box)

        self.volume_box = QGroupBox("Volumetric Mode: Volume")
        vl = QVBoxLayout(self.volume_box)
        self.inputs["VOLUME"] = LabeledSlider("Volume (m³)", 100, 1000000, 5000, 0.1, 4)
        vl.addWidget(self.inputs["VOLUME"])
        layout.addWidget(self.volume_box)
        layout.addStretch()

    def setup_aerostatics_tab(self):
        self.aerostatics_tab = QWidget()
        layout = QVBoxLayout(self.aerostatics_tab)

        # 1. Atmospheric Conditions
        env_grp = QGroupBox("Atmospheric Conditions")
        el = QGridLayout(env_grp)
        self.inputs["OPERATIONAL_HEIGHT"] = LabeledSlider("Op. Altitude (m)", 0, 20000, 4500, 10, 0)
        self.inputs["RELATIVE_HUMIDITY"] = LabeledSlider("Rel. Humidity (0-1)", 0, 1, 0.7, 0.01, 2)
        self.inputs["MARGIN_HEIGHT"] = LabeledSlider("Pressure Margin (m)", 0, 2000, 500, 10, 0)
        el.addWidget(self.inputs["OPERATIONAL_HEIGHT"], 0, 0)
        el.addWidget(self.inputs["RELATIVE_HUMIDITY"], 0, 1)
        el.addWidget(self.inputs["MARGIN_HEIGHT"], 1, 0)
        layout.addWidget(env_grp)

        # 2. Lifting Gas Properties
        gas_grp = QGroupBox("Lifting Gas Properties")
        gl = QGridLayout(gas_grp)
        self.inputs["GAS_PURITY"] = LabeledSlider("Purity (0-1)", 0.8, 1, 0.97, 0.001, 3)
        self.inputs["GAS_CONSTANT"] = LabeledSlider("Gas Constant (He=2077)", 200, 4200, 2077, 1, 0)
        gl.addWidget(self.inputs["GAS_PURITY"], 0, 0)
        gl.addWidget(self.inputs["GAS_CONSTANT"], 0, 1)
        layout.addWidget(gas_grp)

        # 3. Mass and Structural Design
        mass_grp = QGroupBox("Mass and Structural Design")
        ml = QGridLayout(mass_grp)
        self.inputs["SKIN_DENSITY"] = LabeledSlider("Skin Density (kg/m²)", 0.1, 2.0, 0.75, 0.01, 2)
        self.inputs["PAYLOAD_MASS"] = LabeledSlider("Payload/Add. Mass (kg)", 0, 5000, 220, 1, 1)

        # Tether Controls
        self.inputs["INCLUDE_TETHER"] = QCheckBox("INCLUDE TETHER MASS")
        self.inputs["INCLUDE_TETHER"].setChecked(True)
        self.inputs["INCLUDE_TETHER"].setStyleSheet("color: #00BFFF; font-weight: bold;")

        # NEW: Connect the checkbox to the toggle function
        self.inputs["INCLUDE_TETHER"].toggled.connect(self._toggle_tether_inputs)

        self.inputs["TETHER_DENSITY"] = LabeledSlider("Tether Density (kg/m)", 0, 5, 0.1, 0.01, 2)
        self.inputs["TETHER_FRACTION"] = LabeledSlider("Tether Fraction (0-1)", 0, 1, 1.0, 0.01, 2)

        self.inputs["TARGET_NET_LIFT"] = LabeledSlider("Target Net Lift (N)", -1000, 5000, 0, 1, 1)
        self.inputs["OPTIMIZE_LENGTH"] = QCheckBox("OPTIMIZE LENGTH FOR TARGET LIFT")
        self.inputs["OPTIMIZE_LENGTH"].setChecked(True)

        ml.addWidget(self.inputs["SKIN_DENSITY"], 0, 0)
        ml.addWidget(self.inputs["PAYLOAD_MASS"], 0, 1)
        ml.addWidget(self.inputs["INCLUDE_TETHER"], 1, 0)
        ml.addWidget(self.inputs["TETHER_DENSITY"], 2, 0)
        ml.addWidget(self.inputs["TETHER_FRACTION"], 2, 1)
        ml.addWidget(self.inputs["TARGET_NET_LIFT"], 3, 0)
        ml.addWidget(self.inputs["OPTIMIZE_LENGTH"], 4, 0, 1, 2)
        layout.addWidget(mass_grp)

        layout.addStretch()

    def _toggle_tether_inputs(self, enabled):
        """Enables or disables tether-related sliders based on the checkbox state."""
        self.inputs["TETHER_DENSITY"].setEnabled(enabled)
        self.inputs["TETHER_FRACTION"].setEnabled(enabled)

    def refresh_tabs(self):
        self.tab_widget.blockSignals(True)
        mode_id = self.mode_button_group.checkedId()

        is_standard = (mode_id == 1)
        is_vol = (mode_id == 2)
        is_balloon = (mode_id == 3)
        is_aero = (mode_id == 4)
        is_multi = self.lobe_button_group.checkedId() > 1 and not is_balloon

        # Toggle Input group visibility
        self.hull_shape_box.setHidden(is_balloon)
        self.balloon_params_box.setHidden(not is_balloon)
        self.hull_config_group.setHidden(is_balloon)
        self.length_box.setHidden(not is_standard)
        self.volume_box.setHidden(not (is_vol or is_balloon))

        # Handle Output Tab UI rearrangement
        if is_aero:
            self.btn_run.setText("CALCULATE PERFORMANCE")
            self.btn_run.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
            self.btn_plot.hide()
            self.btn_csv.show()

            # Hide Added Mass and Export Format controls in Aerostatic mode
            self.inputs["COMPUTE_ADDED_MASS"].hide()
            # To hide the radio buttons, we hide the layout container they live in
            for i in range(self.format_button_group.buttons().__len__()):
                self.format_button_group.button(i).hide()

            self.matrix_group.hide()
            self.preview_group.hide()
            self.aero_analysis_group.show()
        else:
            self.btn_run.setText("RUN GENERATION")
            self.btn_run.setStyleSheet("background-color: #007ACC; color: white;")
            self.btn_plot.show()
            self.btn_csv.hide()

            # Show Added Mass and Export Format controls in other modes
            self.inputs["COMPUTE_ADDED_MASS"].show()
            for i in range(self.format_button_group.buttons().__len__()):
                self.format_button_group.button(i).show()

            self.matrix_group.show()
            self.preview_group.show()
            self.aero_analysis_group.hide()

        # Update available tabs
        curr = self.tab_widget.currentIndex()
        self.tab_widget.clear()
        self.tab_widget.addTab(self.primary_input_tab, "Envelope Geometry")

        if is_aero:
            self.tab_widget.addTab(self.aerostatics_tab, "Aerostatic Analysis")
        if is_multi:
            self.tab_widget.addTab(self.fairings_tab, "Multi-Lobe Configuration")
        if not is_balloon:
            self.tab_widget.addTab(self.fin_tab, "Fin Design")

        self.tab_widget.addTab(self.output_tab, "Output")
        self.tab_widget.setCurrentIndex(min(curr, self.tab_widget.count()-1))
        self.tab_widget.blockSignals(False)
        self._update_navigation_buttons()

    def setup_fairings_tab(self):
        layout = QVBoxLayout(self.fairings_tab)

        self.offset_box = QGroupBox("Lobe Separation Offsets")
        ol = QVBoxLayout(self.offset_box)
        self.inputs["LOBE_OFFSET_X_SLIDER"] = LabeledSlider("X Offset (Longitudinal)", 0, 50, 10, 0.0001, 4)
        self.inputs["LOBE_OFFSET_Y_SLIDER"] = LabeledSlider("Y Offset (Lateral)", 0, 50, 10, 0.0001, 4)
        self.inputs["LOBE_OFFSET_Z_SLIDER"] = LabeledSlider("Z Offset (Vertical)", 0, 50, 10, 0.0001, 4)
        ol.addWidget(self.inputs["LOBE_OFFSET_X_SLIDER"])
        ol.addWidget(self.inputs["LOBE_OFFSET_Y_SLIDER"])
        ol.addWidget(self.inputs["LOBE_OFFSET_Z_SLIDER"])
        layout.addWidget(self.offset_box)

        sheet_grp = QGroupBox("Fairing Geometry")
        sl = QVBoxLayout(sheet_grp)
        self.inputs["SHEET_LENGTH_RATIO_SLIDER"] = LabeledSlider("Sheet Length Ratio (0-1)", 0, 1, 0.5, 0.0001, 4)
        sl.addWidget(self.inputs["SHEET_LENGTH_RATIO_SLIDER"])
        layout.addWidget(sheet_grp)
        layout.addStretch()

    def setup_fin_tab(self):
        main_layout = QVBoxLayout(self.fin_tab)
        main_layout.setContentsMargins(15, 15, 15, 15)

        self.inputs["INCLUDE_FINS"] = QCheckBox("GENERATE FINS WITH HULL")
        self.inputs["INCLUDE_FINS"].setObjectName("FinToggle")
        self.inputs["INCLUDE_FINS"].setChecked(True)
        self.inputs["INCLUDE_FINS"].toggled.connect(self._toggle_fin_inputs)
        main_layout.addWidget(self.inputs["INCLUDE_FINS"])

        self.fin_container = QWidget()
        container_layout = QVBoxLayout(self.fin_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        fin_dim_group = QGroupBox("Fin Dimensions")
        fin_dim_layout = QGridLayout(fin_dim_group)
        self.inputs["FIN_RC_LENGTH"] = LabeledSlider("Root Chord Length", 5.0, 50.0, 15.5, 0.0001, 4)
        self.inputs["FIN_HEIGHT"] = LabeledSlider("Fin Height (Span)", 5.0, 50.0, 15.5, 0.0001, 4)
        self.inputs["FIN_THICKNESS"] = LabeledSlider("Thickness %", 1.0, 20.0, 10.0, 0.0001, 4)
        self.inputs["FIN_TAPER_RATIO"] = LabeledSlider("Taper Ratio", 0.1, 1.0, 0.55, 0.0001, 4)
        self.inputs["FIN_AXIAL_OFFSET"] = LabeledSlider("Axial Offset %", 50.0, 100.0, 80.0, 0.0001, 4)
        self.inputs["FIN_SECTION_RESOLUTION"] = LabeledSlider("Section Resolution", 10, 100, 60, 1, decimals=0)

        fin_dim_layout.addWidget(self.inputs["FIN_RC_LENGTH"], 0, 0)
        fin_dim_layout.addWidget(self.inputs["FIN_HEIGHT"], 0, 1)
        fin_dim_layout.addWidget(self.inputs["FIN_THICKNESS"], 0, 2)
        fin_dim_layout.addWidget(self.inputs["FIN_TAPER_RATIO"], 1, 0)
        fin_dim_layout.addWidget(self.inputs["FIN_AXIAL_OFFSET"], 1, 1)
        fin_dim_layout.addWidget(self.inputs["FIN_SECTION_RESOLUTION"], 1, 2)
        container_layout.addWidget(fin_dim_group)

        fin_sweep_group = QGroupBox("Fin Sweep and Configuration")
        fin_sweep_layout = QGridLayout(fin_sweep_group)
        self.inputs["FIN_SWEEP_ANGLE"] = LabeledSlider("Sweep Angle (Deg)", 0.0, 45.0, 0.0, 0.0001, 4)
        self.inputs["FIN_TIP_ANGLE"] = LabeledSlider("Tip Angle (Deg)", 0.0, 30.0, 15.0, 0.0001, 4)
        self.inputs["FIN_NUMBER"] = LabeledSlider("N Fins", 2, 8, 4, 1, decimals=0)

        fin_sweep_layout.addWidget(self.inputs["FIN_SWEEP_ANGLE"], 0, 0)
        fin_sweep_layout.addWidget(self.inputs["FIN_TIP_ANGLE"], 0, 1)
        fin_sweep_layout.addWidget(QLabel("Number of Fins:"), 1, 0)
        fin_sweep_layout.addWidget(self.inputs["FIN_NUMBER"], 1, 1)
        fin_sweep_layout.addWidget(QLabel("Angular Positions:"), 2, 0)
        self.inputs["FIN_THETA_POS_TEXT"] = QLineEdit("0.0, 90.0, 180.0, 270.0")
        fin_sweep_layout.addWidget(self.inputs["FIN_THETA_POS_TEXT"], 2, 1, 1, 2)
        container_layout.addWidget(fin_sweep_group)

        main_layout.addWidget(self.fin_container)
        main_layout.addStretch(1)

    def _toggle_fin_inputs(self, enabled):
        self.fin_container.setEnabled(enabled)

    def setup_output_tab(self):
        self.splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.inputs["FINAL_OBJECT_NAME"] = QLineEdit("Airship_Project")
        left_layout.addWidget(QLabel("Project Name:"))
        left_layout.addWidget(self.inputs["FINAL_OBJECT_NAME"])

        dir_group = QGroupBox("Base Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        self.dir_path_display = QLineEdit(self.base_output_directory)
        self.dir_path_display.setReadOnly(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_output_directory)
        dir_layout.addWidget(self.dir_path_display)
        dir_layout.addWidget(btn_browse)
        left_layout.addWidget(dir_group)

        self.inputs["N_PETALS"] = LabeledSlider("Gores/Petals", 2, 200, 8, 1, 0)
        left_layout.addWidget(self.inputs["N_PETALS"])

        self.inputs["COMPUTE_ADDED_MASS"] = QCheckBox("COMPUTE ADDED MASS (May slow down process)")
        self.inputs["COMPUTE_ADDED_MASS"].setChecked(True)
        self.inputs["COMPUTE_ADDED_MASS"].setFont(QFont("Arial", 9, QFont.Bold))
        left_layout.addWidget(self.inputs["COMPUTE_ADDED_MASS"])

        self.format_button_group = QButtonGroup(self)
        h_lay = QHBoxLayout()
        for i, fmt in enumerate([".brep", ".stl", ".step"]):
            btn = QPushButton(fmt)
            btn.setCheckable(True)
            h_lay.addWidget(btn)
            self.format_button_group.addButton(btn, i)
            btn.setChecked(i==1)
        left_layout.addLayout(h_lay)

        btn_lay = QHBoxLayout()
        self.btn_run = QPushButton("RUN GENERATION")
        self.btn_run.setMinimumHeight(35)
        self.btn_run.setStyleSheet("background-color: #007ACC; color: white;")
        self.btn_run.clicked.connect(self.handle_output_action)

        self.btn_plot = QPushButton("PLOT 2D PETAL")
        self.btn_plot.setMinimumHeight(35)
        self.btn_plot.clicked.connect(self.generate_plot)

        self.btn_csv = QPushButton("EXPORT CSV")
        self.btn_csv.setMinimumHeight(35)
        self.btn_csv.setStyleSheet("background-color: #555555; color: white;")
        self.btn_csv.clicked.connect(self.export_csv_data)
        self.btn_csv.hide()

        btn_lay.addWidget(self.btn_run)
        btn_lay.addWidget(self.btn_plot)
        btn_lay.addWidget(self.btn_csv)
        left_layout.addLayout(btn_lay)

        prop_group = QGroupBox("Geometric Properties")
        prop_layout = QGridLayout(prop_group)
        self.prop_outputs = {}
        labels = ["Vol (m³):", "Surf (m²):", "Top (m²):", "Side (m²):"]
        keys = ["vol", "surf", "top_area", "side_area"]

        for i, (lbl, key) in enumerate(zip(labels, keys)):
            prop_layout.addWidget(QLabel(lbl), i // 2, (i % 2) * 2)
            self.prop_outputs[key] = QLineEdit("0.0000")
            self.prop_outputs[key].setReadOnly(True)
            self.prop_outputs[key].setStyleSheet("background-color: #2D2D2D; color: #00BFFF; font-weight: bold;")
            prop_layout.addWidget(self.prop_outputs[key], i // 2, (i % 2) * 2 + 1)
        left_layout.addWidget(prop_group)

        self.log = QTextEdit("Status: Ready")
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(200)
        left_layout.addWidget(self.log)

        right_widget = QWidget()
        self.right_layout = QVBoxLayout(right_widget)

        self.matrix_group = QGroupBox("Added Mass Matrix")
        matrix_vbox = QVBoxLayout(self.matrix_group)
        self.matrix_table = QTableWidget(6, 6)
        self.matrix_table.setHorizontalHeaderLabels(["u", "v", "w", "p", "q", "r"])
        self.matrix_table.setVerticalHeaderLabels(["X", "Y", "Z", "K", "M", "N"])
        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.matrix_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.matrix_table.setFixedHeight(180)
        self.matrix_table.setEditTriggers(QTableWidget.NoEditTriggers)
        matrix_vbox.addWidget(self.matrix_table)
        self.right_layout.addWidget(self.matrix_group)

        self.aero_analysis_group = QGroupBox("Aerostatic Performance Analysis")
        self.aero_analysis_group.hide()
        aero_vbox = QVBoxLayout(self.aero_analysis_group)
        self.fig = Figure(facecolor='#1e1e1e', tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        aero_vbox.addWidget(self.canvas)
        self.right_layout.addWidget(self.aero_analysis_group)

        self.preview_group = QGroupBox("3D Model Preview")
        preview_vbox = QVBoxLayout(self.preview_group)
        self.plotter = BackgroundPlotter(show=False)
        self.plotter.set_background("#1e1e1e")
        preview_vbox.addWidget(self.plotter.interactor)
        self.right_layout.addWidget(self.preview_group)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        main_tab_layout = QVBoxLayout(self.output_tab)
        main_tab_layout.addWidget(self.splitter)

        geo_keys = ["l2d", "m1", "r0", "r1", "cp", "ENVELOPE_LENGTH", "VOLUME"]
        for key in geo_keys:
            if key in self.inputs:
                self.inputs[key].value_changed_by_user.connect(self._auto_update_props)
        self.preset_combo.currentIndexChanged.connect(self._auto_update_props)

    def handle_output_action(self):
        """Routes the button click based on mode; skips generation in Aero mode."""
        if self.mode_button_group.checkedId() == 4:
            self.run_instant_aerostatics()
        else:
            self.run_process()

    def run_instant_aerostatics(self):
        """Performs analytical performance calculations (Bypassing Geometry Export)."""
        self.log.append("[PROCESS] Running Analytical Aerostatic Solver...")
        try:
            target_dir = self.current_session_folder
            p = self.get_parameters(target_dir)

            from aerostatics import AerostatHull
            from geometry_handler import GertlerEnvelope

            resolved_env = GertlerEnvelope.from_parameters(p["ENVELOPE_PARAMS"], p["ENVELOPE_LENGTH"])

            ahull = AerostatHull(
                envelope=resolved_env,
                additional_mass=p.get("PAYLOAD_MASS", 220),
                skin_density=p.get("SKIN_DENSITY", 0.75),
                operational_height=p.get("OPERATIONAL_HEIGHT", 4500),
                deployment_height=0,
                margin_height=500,
                RH=p.get("RELATIVE_HUMIDITY", 0.7),
                purity=p.get("GAS_PURITY", 0.97),
                delta_P=p.get("DELTA_P", 500),
                delta_T=p.get("DELTA_T", 5),
                gas_constant=p.get("GAS_CONSTANT", 2077),
                lobe_number=p.get("LOBE_NUMBER", 1),
                e=p.get("LOBE_OFFSET_X", 0),
                f=p.get("LOBE_OFFSET_Y", 0),
                g=p.get("LOBE_OFFSET_Z", 0),
                # Pass tether data
                tether_density=p.get("TETHER_DENSITY", 0),
                tether_fraction=p.get("TETHER_FRACTION", 1.0)
            )

            # Use the include_tether flag in properties calculation
            h, Ln, Lg, I, BV = ahull.get_properties(n=100, include_tether=p.get("INCLUDE_TETHER"))
            self.last_aero_data = (h, Ln, Lg, I, BV)
            self.update_aero_plots(h, Ln, Lg, I, BV)

            self.log.append(f"[SUCCESS] Solver complete. Hull Length: {p['ENVELOPE_LENGTH']:.3f} m")
        except Exception as e:
            self.log.append(f"[ERROR] Solver failed: {str(e)}")

    def _auto_update_props(self):
        """Refreshes geometric property labels based on current slider states."""
        if self.mode_button_group.checkedId() == 3: # Balloon Mode
            for key in self.prop_outputs:
                self.prop_outputs[key].setText("0.0000")
            return

        params = self.get_parameters(self.current_session_folder)
        if params:
            try:
                from geometry import AirshipGeometry
                geom = AirshipGeometry(params, self.salome_path)
                vol, surf, top, side = geom.geometric_properties()
                self.prop_outputs["vol"].setText(f"{vol:.4f}")
                self.prop_outputs["surf"].setText(f"{surf:.4f}")
                self.prop_outputs["top_area"].setText(f"{top:.4f}")
                self.prop_outputs["side_area"].setText(f"{side:.4f}")
            except Exception:
                pass

    def _update_property_display(self, params):
        """Directly updates the UI text fields with calculated geometric values."""
        try:
            from geometry import AirshipGeometry
            geom = AirshipGeometry(params, self.salome_path)
            vol, surf, top, side = geom.geometric_properties()

            self.prop_outputs["vol"].setText(f"{vol:.4f}")
            self.prop_outputs["surf"].setText(f"{surf:.4f}")
            self.prop_outputs["top_area"].setText(f"{top:.4f}")
            self.prop_outputs["side_area"].setText(f"{side:.4f}")
        except Exception:
            pass

    def _update_3d_view(self, stl_path):
        """Refreshes the 3D model in the plotter interactor."""
        if not os.path.exists(stl_path):
            self.plotter.clear()
            return
        try:
            self.plotter.clear()
            mesh = pv.read(stl_path)
            self.plotter.add_mesh(mesh, color="#00BFFF", show_edges=True, edge_color="#333333", opacity=1)
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        except Exception as e:
            print(f"[ERROR] 3D View failed: {e}")

    def browse_output_directory(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Base Output Directory", self.base_output_directory)
        if selected_dir:
            self.base_output_directory = selected_dir
            self.dir_path_display.setText(selected_dir)
            self.current_session_folder = selected_dir

    def create_new_output_folder(self):
        if not os.path.exists(self.base_output_directory):
            os.makedirs(self.base_output_directory)

        existing_items = os.listdir(self.base_output_directory)
        indices = []
        for item in existing_items:
            if os.path.isdir(os.path.join(self.base_output_directory, item)):
                match = re.match(r"Output_(\d+)", item)
                if match:
                    indices.append(int(match.group(1)))

        next_idx = max(indices) + 1 if indices else 1
        new_folder = os.path.join(self.base_output_directory, f"Output_{next_idx}")
        os.makedirs(new_folder)
        self.current_session_folder = new_folder
        return new_folder

    def export_csv_data(self):
        """Exports calculation results to a CSV file."""
        if not hasattr(self, 'last_aero_data') or self.last_aero_data is None:
            QMessageBox.warning(self, "No Data", "Run 'Calculate' first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Performance Data", "", "CSV Files (*.csv)")
        if path:
            import csv
            h, Ln, Lg, I, BV = self.last_aero_data
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Alt (m)", "Net Lift (N)", "Gross Lift (N)", "Inflation (%)", "Ballonet (m3)"])
                    for row in zip(h, Ln, Lg, I, BV):
                        writer.writerow(row)
                self.log.append(f"[SUCCESS] Exported to: {path}")
            except Exception as e:
                self.log.append(f"[ERROR] CSV Export failed: {e}")

    def get_parameters(self, target_dir):
        """
        Gathers all current GUI inputs into a single parameters dictionary.
        Handles analytical optimization for Aerostatic mode and length scaling for Volumetric mode.
        """
        p = {}
        # Comprehensive list of all slider keys used across all tabs, updated with Tether parameters
        all_keys = [
            "ENVELOPE_LENGTH", "ENVELOPE_RESOLUTION", "m1", "r0", "r1", "cp", "l2d",
            "FIN_AXIAL_OFFSET", "FIN_RC_LENGTH", "FIN_HEIGHT", "FIN_THICKNESS",
            "FIN_TAPER_RATIO", "FIN_SWEEP_ANGLE", "FIN_TIP_ANGLE", "FIN_NUMBER",
            "FIN_SECTION_RESOLUTION", "THETA_RES", "PHI_RES", "ASPECT_RATIO",
            "BULGE_AMPLITUDE", "BULGE_POWER", "GORE_AMPLITUDE", "GORE_FADE_POWER",
            "OPERATIONAL_HEIGHT", "RELATIVE_HUMIDITY", "GAS_PURITY", "GAS_CONSTANT",
            "DELTA_P", "DELTA_T", "SKIN_DENSITY", "PAYLOAD_MASS", "TARGET_NET_LIFT",
            "TETHER_DENSITY", "TETHER_FRACTION"  # NEW: Added for optional tether logic
        ]

        # Extract numerical values from sliders
        for key in all_keys:
            if key in self.inputs:
                p[key] = self.inputs[key].get_value()

        # Extract fixed/text, multi-lobe inputs, and the new tether toggle
        p["INCLUDE_TETHER"] = self.inputs["INCLUDE_TETHER"].isChecked() # NEW: Optionality toggle
        p["N_PETALS"] = self.inputs["N_PETALS"].get_value()
        p["LOBE_OFFSET_X"] = self.inputs["LOBE_OFFSET_X_SLIDER"].get_value()
        p["LOBE_OFFSET_Y"] = self.inputs["LOBE_OFFSET_Y_SLIDER"].get_value()
        p["LOBE_OFFSET_Z"] = self.inputs["LOBE_OFFSET_Z_SLIDER"].get_value()
        p["SHEET_LENGTH_RATIO"] = self.inputs["SHEET_LENGTH_RATIO_SLIDER"].get_value()
        p["LOBE_NUMBER"] = self.lobe_button_group.checkedId()
        p["FINAL_OBJECT_NAME"] = self.inputs["FINAL_OBJECT_NAME"].text()
        p["INCLUDE_FINS"] = self.inputs["INCLUDE_FINS"].isChecked()
        p["ENVELOPE_PARAMS"] = (p["m1"], p["r0"], p["r1"], p["cp"], p["l2d"])

        mode_id = self.mode_button_group.checkedId()

        # --- AEROSTATIC OPTIMIZATION LOGIC (MODE 4) ---
        if mode_id == 4:
            from aerostatics import AerostatHull
            from geometry_handler import GertlerEnvelope

            # Only run optimization if the specific UI checkbox is enabled
            if self.inputs.get("OPTIMIZE_LENGTH") and self.inputs["OPTIMIZE_LENGTH"].isChecked():
                try:
                    # Initialize temporary hull with unit length to solve for target lift
                    base_env = GertlerEnvelope.from_parameters(p["ENVELOPE_PARAMS"], 1.0)

                    ahull = AerostatHull(
                        envelope=base_env,
                        additional_mass=p.get("PAYLOAD_MASS", 220),
                        skin_density=p.get("SKIN_DENSITY", 0.75),
                        operational_height=p.get("OPERATIONAL_HEIGHT", 4500),
                        deployment_height=0,
                        margin_height=500,
                        RH=p.get("RELATIVE_HUMIDITY", 0.7),
                        purity=p.get("GAS_PURITY", 0.97),
                        delta_P=p.get("DELTA_P", 500),
                        delta_T=p.get("DELTA_T", 5),
                        gas_constant=p.get("GAS_CONSTANT", 2077),
                        lobe_number=p.get("LOBE_NUMBER", 1),
                        e=p.get("LOBE_OFFSET_X", 0),
                        f=p.get("LOBE_OFFSET_Y", 0),
                        g=p.get("LOBE_OFFSET_Z", 0),
                        # Passing tether parameters to the solver
                        tether_density=p.get("TETHER_DENSITY", 0) if p["INCLUDE_TETHER"] else 0,
                        tether_fraction=p.get("TETHER_FRACTION", 1.0),
                        fin_rc=p.get("FIN_RC_LENGTH", 0),
                        fin_height=p.get("FIN_HEIGHT", 0),
                        fin_taper_ratio=p.get("FIN_TAPER_RATIO", 1),
                        fin_thickness=p.get("FIN_THICKNESS", 0),
                        fin_number=p.get("FIN_NUMBER", 4)
                    )

                    # Find length required to achieve TARGET_NET_LIFT
                    target_lift = p.get("TARGET_NET_LIFT", 0)
                    resolved_env, _ = ahull.initialise_from_operational_altitude([1.0, 1000.0], target_lift=target_lift)

                    # Update parameter dictionary and UI slider with the result
                    p["ENVELOPE_LENGTH"] = resolved_env.length
                    self.inputs["ENVELOPE_LENGTH"].set_value(resolved_env.length)

                except Exception as e:
                    print(f"[PROCESS] Aerostatic optimization failed: {e}")
                    p["ENVELOPE_LENGTH"] = self.inputs["ENVELOPE_LENGTH"].get_value()
            else:
                # If optimization is unchecked, use the user-defined slider length
                p["ENVELOPE_LENGTH"] = self.inputs["ENVELOPE_LENGTH"].get_value()

        # --- VOLUMETRIC MODE LOGIC (MODE 2) ---
        elif mode_id == 2:
            from geometry_handler import GertlerEnvelope
            # Calculate length based on a specific target volume
            temp_env = GertlerEnvelope.from_parameters_volume(
                p["ENVELOPE_PARAMS"], self.inputs["VOLUME"].get_value(),
                int(p["ENVELOPE_RESOLUTION"]), p["LOBE_NUMBER"],
                p["LOBE_OFFSET_X"], p["LOBE_OFFSET_Y"], p["LOBE_OFFSET_Z"]
            )
            p["ENVELOPE_LENGTH"] = temp_env.length
            self.inputs["ENVELOPE_LENGTH"].set_value(temp_env.length)

        # Finalize Fin Offset based on the final determined length
        hull_len = p["ENVELOPE_LENGTH"]
        req_le = (p.get("FIN_AXIAL_OFFSET", 80) / 100.0) * hull_len
        max_le = hull_len - p.get("FIN_RC_LENGTH", 15) - 0.5
        p["FIN_AXIAL_OFFSET"] = min(req_le, max_le)

        # Parse Fin angular positions
        try:
            theta_text = self.inputs["FIN_THETA_POS_TEXT"].text()
            p["FIN_THETA_POS"] = [float(a.strip()) for a in theta_text.split(',') if a.strip()]
        except:
            p["FIN_THETA_POS"] = [0, 90, 180, 270]

        # Final assembly of directory and specialized balloon parameters
        p["OUTPUT_DIRECTORY"] = target_dir
        p["SALOME_PATH"] = self.salome_path
        p["balloon_params"] = {
            "ASPECT_RATIO": p.get("ASPECT_RATIO"),
            "GORE_AMPLITUDE": p.get("GORE_AMPLITUDE"),
            "BULGE_AMPLITUDE": p.get("BULGE_AMPLITUDE"),
            "BULGE_POWER": p.get("BULGE_POWER"),
            "GORE_FADE": p.get("GORE_FADE_POWER")
        }
        return p

    def run_process(self):
        target_dir = self.create_new_output_folder()
        p = self.get_parameters(target_dir)
        if p is None: return

        fmt_idx = self.format_button_group.checkedId()
        p['EXPORT_FORMAT'] = ["BREP", "STL", "STEP"][fmt_idx]
        p['SALOME_PATH'] = self.salome_path

        # UI FEEDBACK
        self.btn_run.setEnabled(False)
        self.btn_run.setText("PROCESSING...")
        self.log.append("\n[GUI] Starting generation process...")

        # START THREAD
        self.thread = QThread()
        self.worker = GenerationWorker(
            self.mode_button_group.checkedId(),
            p,
            self.inputs["VOLUME"].get_value(),
            self.inputs["GORE_MODEL"].currentText(),
            compute_added_mass=self.inputs["COMPUTE_ADDED_MASS"].isChecked()
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def update_aero_plots(self, h, Ln, Lg, I, BV):
        """Updates the performance graphs with native dark mode styling."""
        self.fig.clear()

        titles = ["Net Static Lift (N)", "Gross Static Lift (N)", "Inflation Fraction (%)", "Ballonet Volume (m³)"]
        datasets = [Ln, Lg, I * 100, BV]
        colors = ['#00BFFF', '#00FF00', '#FF4500', '#DA70D6']

        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i + 1)
            ax.plot(h, datasets[i], color=colors[i], linewidth=1.5)
            ax.set_title(titles[i], color='#00BFFF', fontsize=10, fontweight='bold')
            ax.set_xlabel("Altitude (m)", color='white', fontsize=8)
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, alpha=0.1, linestyle='--')
            ax.set_facecolor('#1e1e1e')
            for spine in ax.spines.values():
                spine.set_color('#3c3c3c')

        self.fig.tight_layout()
        self.canvas.draw()

    def on_worker_finished(self, result):
        matrix, stl_path = result
        self.btn_run.setEnabled(True)
        self.btn_run.setText("RUN GENERATION")

        if os.path.exists(stl_path):
            self.plotter.clear()
            mesh = pv.read(stl_path)
            self.plotter.add_mesh(mesh, color="#00BFFF", show_edges=True, edge_color="#333333", opacity=0.8)
            self.plotter.reset_camera()

        if self.mode_button_group.checkedId() == 4:
            try:
                p = self.get_parameters(self.current_session_folder)
                from aerostatics import AerostatHull
                from geometry_handler import GertlerEnvelope

                resolved_env = GertlerEnvelope.from_parameters(p["ENVELOPE_PARAMS"], p["ENVELOPE_LENGTH"])
                # FIXED: Passing full parameters to match the updated __init__ and get_parameters
                ahull = AerostatHull(
                    envelope=resolved_env,
                    additional_mass=p.get("PAYLOAD_MASS", 220),
                    skin_density=p.get("SKIN_DENSITY", 0.75),
                    operational_height=p.get("OPERATIONAL_HEIGHT", 4500),
                    deployment_height=0,
                    margin_height=500,
                    RH=p.get("RELATIVE_HUMIDITY", 0.7),
                    purity=p.get("GAS_PURITY", 0.97),
                    delta_P=p.get("DELTA_P", 500),
                    delta_T=p.get("DELTA_T", 5),
                    gas_constant=p.get("GAS_CONSTANT", 2077),
                    lobe_number=p.get("LOBE_NUMBER", 1),
                    e=p.get("LOBE_OFFSET_X", 0),
                    f=p.get("LOBE_OFFSET_Y", 0),
                    g=p.get("LOBE_OFFSET_Z", 0),
                    fin_rc=p.get("FIN_RC_LENGTH", 0),
                    fin_height=p.get("FIN_HEIGHT", 0),
                    fin_taper_ratio=p.get("FIN_TAPER_RATIO", 1),
                    fin_thickness=p.get("FIN_THICKNESS", 0),
                    fin_number=p.get("FIN_NUMBER", 4)
                )

                h, Ln, Lg, I, BV = ahull.get_properties(n=100)
                self.update_aero_plots(h, Ln, Lg, I, BV)
                self.log.append(f"[SUCCESS] Dashboard updated for {p['ENVELOPE_LENGTH']:.3f} m hull.")
            except Exception as e:
                self.log.append(f"[ERROR] Aerostatic display failed: {str(e)}")

        elif matrix is not None:
            for r in range(6):
                for c in range(6):
                    self.matrix_table.setItem(r, c, QTableWidgetItem(f"{matrix[r,c]:.4f}"))

        self.log.append("[GUI] Process successfully completed.")

    def on_worker_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("RUN GENERATION")
        QMessageBox.critical(self, "Error", f"Failed: {error_msg}")

    def generate_plot(self):
        target_dir = self.current_session_folder
        p = self.get_parameters(target_dir)
        if p is None: return
        try:
            dat_file = os.path.join(target_dir, f"{p['FINAL_OBJECT_NAME']}.dat")
            msg = plot_and_save_profile(
                p["ENVELOPE_PARAMS"],
                p["ENVELOPE_LENGTH"],
                int(p["ENVELOPE_RESOLUTION"]),
                int(p["N_PETALS"]),
                int(p["ENVELOPE_RESOLUTION"]),
                dat_file,
                p["FINAL_OBJECT_NAME"]
            )
            print(f"[GUI] Plotting results: {msg}")
        except Exception as e:
            print(f"Plot Error: {e}")

    def load_defaults(self):
        self.load_preset(0)

    def load_preset(self, idx):
        vals = STANDARD_ENVELOPES[self.preset_combo.itemText(idx)]
        for k, v in zip(["m1", "r0", "r1", "cp", "l2d"], vals):
            if k in self.inputs: self.inputs[k].set_value(v)

    def reset_to_defaults(self):
        # 1. Gertler Geometry Sliders (Based on current dropdown preset)
        current_preset = self.preset_combo.currentText()
        vals = STANDARD_ENVELOPES[current_preset]
        for k, v in zip(["m1", "r0", "r1", "cp", "l2d"], vals):
            if k in self.inputs: self.inputs[k].set_value(v)

        # 2. General Dimensions
        self.inputs["ENVELOPE_LENGTH"].set_value(100.0)
        self.inputs["VOLUME"].set_value(5000.0)
        self.inputs["ENVELOPE_RESOLUTION"].set_value(150)
        self.inputs["N_PETALS"].set_value(8)

        # 3. Super Pressure Balloon Sliders (Mode 3)
        balloon_keys = {
            "ASPECT_RATIO": 1.0, "GORE_AMPLITUDE": 0.05,
            "THETA_RES": 400, "PHI_RES": 600,
            "BULGE_AMPLITUDE": 0.0, "BULGE_POWER": 1.0, "GORE_FADE_POWER": 4.0
        }
        for key, val in balloon_keys.items():
            if key in self.inputs: self.inputs[key].set_value(val)

        # 4. Aerostatic Sliders (Mode 4)
        aero_keys = {
            "OPERATIONAL_HEIGHT": 4500.0, "RELATIVE_HUMIDITY": 0.7,
            "GAS_PURITY": 0.97, "GAS_CONSTANT": 2077.0,
            "DELTA_P": 500.0, "DELTA_T": 5.0,
            "SKIN_DENSITY": 0.75, "PAYLOAD_MASS": 220.0, "TARGET_NET_LIFT": 0.0
        }
        for key, val in aero_keys.items():
            if key in self.inputs: self.inputs[key].set_value(val)

        # 5. Multi-Lobe & Fin Sliders
        for l_key in ["LOBE_OFFSET_X_SLIDER", "LOBE_OFFSET_Y_SLIDER", "LOBE_OFFSET_Z_SLIDER"]:
            if l_key in self.inputs: self.inputs[l_key].set_value(10.0)

        # 6. Clear Output Displays
        self.log.clear()
        self.log.append("Status: Ready (Sliders Reset)")
        self.matrix_table.clearContents()
        if hasattr(self, 'fig'):
            self.fig.clear()
            self.canvas.draw()
        self.plotter.clear()
        self.inputs["FINAL_OBJECT_NAME"].setText("Airship_Project")

        # Update analytical properties labels
        if hasattr(self, '_auto_update_props'):
            self._auto_update_props()

    def _update_navigation_buttons(self):
        idx = self.tab_widget.currentIndex()
        count = self.tab_widget.count()
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < count - 1)

    def setup_navigation_buttons(self):
        nav = QWidget()
        lay = QHBoxLayout(nav)

        self.btn_back = QPushButton("<< PREV")
        self.btn_back.setMinimumHeight(35)
        self.btn_back.setFixedWidth(100)

        self.btn_next = QPushButton("NEXT >>")
        self.btn_next.setMinimumHeight(35)
        self.btn_next.setFixedWidth(100)
        self.btn_next.setStyleSheet("background-color: #00BFFF; color: black; font-weight: bold;")

        self.btn_back.clicked.connect(lambda: self.tab_widget.setCurrentIndex(self.tab_widget.currentIndex()-1))
        self.btn_next.clicked.connect(lambda: self.tab_widget.setCurrentIndex(self.tab_widget.currentIndex()+1))

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setMinimumHeight(35)
        self.btn_reset.setFixedWidth(80)
        self.btn_reset.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset_to_defaults)

        self.btn_exit = QPushButton("EXIT")
        self.btn_exit.setMinimumHeight(35)
        self.btn_exit.setFixedWidth(80)
        self.btn_exit.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold;")
        self.btn_exit.clicked.connect(self.close)

        lay.addWidget(self.btn_back)
        lay.addWidget(self.btn_next)
        lay.addStretch()
        lay.addWidget(self.btn_reset)
        lay.addWidget(self.btn_exit)
        return nav

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AirshipGUI()
    ex.show()
    sys.exit(app.exec())
