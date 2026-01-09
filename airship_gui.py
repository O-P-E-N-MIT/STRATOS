import sys
import os
import re
import numpy as np
import io

from PySide6.QtGui import QFont, QDoubleValidator
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSlider, QGroupBox, QFileDialog, QTextEdit,
    QButtonGroup, QCheckBox, QMessageBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView
)

# --- 3D VISUALIZATION IMPORTS ---
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from geometry import AirshipGeometry, plot_and_save_profile
from geometry_handler import STANDARD_ENVELOPES
# Integration of balloon.py
from balloon import create_balloon_geometry

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

    def run(self):
        try:
            output_path = os.path.join(self.params['OUTPUT_DIRECTORY'], f"{self.params['FINAL_OBJECT_NAME']}.stl")

            if self.mode_id == 3: # BALLOON MODE
                print(f"[PROCESS] Starting Superpressure Balloon generation...")
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
                print(f"[PROCESS] Launching Salome subprocess for STL export...")
                g = AirshipGeometry(self.params, self.params['SALOME_PATH'])

                # REFINED LOGIC: Pass "STL" string to completely bypass added mass blocks in the geometry scripts
                if not self.compute_added_mass:
                    process, matrix = g.run_salome("STL")
                    matrix = None
                else:
                    print("[PROCESS] Computing Added Mass Matrix...")
                    process, matrix = g.run_salome("FULL")

                print(f"[SUCCESS] Airship STL generated at: {output_path}")
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
        for i, name in enumerate(["STANDARD MODE", "VOLUMETRIC MODE", "SUPER PRESSURE BALLOON"], 1):
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

    def refresh_tabs(self):
        self.tab_widget.blockSignals(True)
        mode_id = self.mode_button_group.checkedId()
        is_vol = (mode_id == 2)
        is_balloon = (mode_id == 3)
        is_multi = self.lobe_button_group.checkedId() > 1 and not is_balloon

        self.hull_shape_box.setHidden(is_balloon)
        self.balloon_params_box.setHidden(not is_balloon)
        self.hull_config_group.setHidden(is_balloon)
        self.length_box.setHidden(is_vol or is_balloon)
        self.volume_box.setHidden(not (is_vol or is_balloon))

        curr = self.tab_widget.currentIndex()
        self.tab_widget.clear()
        self.tab_widget.addTab(self.primary_input_tab, "Envelope Geometry")
        if is_multi: self.tab_widget.addTab(self.fairings_tab, "Multi-Lobe Configuration")
        if not is_balloon: self.tab_widget.addTab(self.fin_tab, "Fin Design")
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
        # Use a Horizontal Splitter for 50/50 division
        self.splitter = QSplitter(Qt.Horizontal)

        # --- LEFT PANEL (Settings, Project Info, and Log) ---
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

        # CHECKBOX: Added Mass calculation toggle
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
        self.btn_run.clicked.connect(self.run_process)

        self.btn_plot = QPushButton("PLOT 2D PETAL")
        self.btn_plot.setMinimumHeight(35)
        self.btn_plot.clicked.connect(self.generate_plot)

        btn_lay.addWidget(self.btn_run)
        btn_lay.addWidget(self.btn_plot)
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

        # --- RIGHT PANEL ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        matrix_group = QGroupBox("Added Mass Matrix")
        matrix_vbox = QVBoxLayout(matrix_group)
        self.matrix_table = QTableWidget(6, 6)
        self.matrix_table.setHorizontalHeaderLabels(["u", "v", "w", "p", "q", "r"])
        self.matrix_table.setVerticalHeaderLabels(["X", "Y", "Z", "K", "M", "N"])
        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.matrix_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.matrix_table.setFixedHeight(180)
        self.matrix_table.setEditTriggers(QTableWidget.NoEditTriggers)
        matrix_vbox.addWidget(self.matrix_table)
        right_layout.addWidget(matrix_group)

        preview_group = QGroupBox("3D Model Preview")
        preview_vbox = QVBoxLayout(preview_group)
        self.plotter = BackgroundPlotter(show=False)
        self.plotter.set_background("#1e1e1e")
        preview_vbox.addWidget(self.plotter.interactor)
        right_layout.addWidget(preview_group)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        main_tab_layout = QVBoxLayout(self.output_tab)
        main_tab_layout.addWidget(self.splitter)

        # --- SIGNAL CONNECTIONS FOR LIVE UPDATES ---
        geo_keys = ["l2d", "m1", "r0", "r1", "cp", "ENVELOPE_LENGTH", "VOLUME"]
        for key in geo_keys:
            if key in self.inputs:
                self.inputs[key].value_changed_by_user.connect(self._auto_update_props)

        # Connect the preset dropdown
        self.preset_combo.currentIndexChanged.connect(self._auto_update_props)

    def _auto_update_props(self):
        """Internal trigger to refresh properties based on current slider states."""
        # Skip for balloon mode if your AirshipGeometry logic doesn't support it
        if self.mode_button_group.checkedId() == 3:
            return

        params = self.get_parameters(self.current_session_folder)
        if params:
            self._update_property_display(params)

    def _update_property_display(self, params):
        """Calculates and updates theoretical properties in the GUI labels."""
        try:
            geom = AirshipGeometry(params, self.salome_path)
            vol, surf, top, side = geom.geometric_properties()

            self.prop_outputs["vol"].setText(f"{vol:.4f}")
            self.prop_outputs["surf"].setText(f"{surf:.4f}")
            self.prop_outputs["top_area"].setText(f"{top:.4f}")
            self.prop_outputs["side_area"].setText(f"{side:.4f}")

        except Exception as e:
            pass

    def _update_3d_view(self, stl_path):
        """Refreshes the 3D model in the plotter interactor."""
        if not os.path.exists(stl_path):
            self.plotter.clear()
            return
        try:
            self.plotter.clear()
            mesh = pv.read(stl_path)
            self.plotter.add_mesh(mesh, color="#00BFFF", show_edges=True, edge_color="#333333", opacity=0.8)
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

    def get_parameters(self, target_dir):
        p = {}
        slider_keys = ["ENVELOPE_LENGTH", "ENVELOPE_RESOLUTION", "m1", "r0", "r1", "cp", "l2d",
                       "FIN_AXIAL_OFFSET", "FIN_RC_LENGTH", "FIN_HEIGHT", "FIN_THICKNESS",
                       "FIN_TAPER_RATIO", "FIN_SWEEP_ANGLE", "FIN_TIP_ANGLE", "FIN_NUMBER",
                       "FIN_SECTION_RESOLUTION"]

        balloon_slider_keys = ["THETA_RES", "PHI_RES", "ASPECT_RATIO", "BULGE_AMPLITUDE",
                               "BULGE_POWER", "GORE_AMPLITUDE", "GORE_FADE_POWER"]

        for key in (slider_keys + balloon_slider_keys):
            if key in self.inputs: p[key] = self.inputs[key].get_value()

        p["N_PETALS"] = self.inputs["N_PETALS"].get_value()
        p["LOBE_OFFSET_X"] = self.inputs["LOBE_OFFSET_X_SLIDER"].get_value()
        p["LOBE_OFFSET_Y"] = self.inputs["LOBE_OFFSET_Y_SLIDER"].get_value()
        p["LOBE_OFFSET_Z"] = self.inputs["LOBE_OFFSET_Z_SLIDER"].get_value()
        p["MULTI_LOBE_OFFSET_FACTOR"] = 0
        p["SHEET_LENGTH_RATIO"] = self.inputs["SHEET_LENGTH_RATIO_SLIDER"].get_value()

        lobe_id = self.lobe_button_group.checkedId()
        p["LOBE_NUMBER"] = lobe_id if lobe_id != -1 else 1
        p["ENVELOPE_PARAMS"] = (p.get("m1", 0.419), p.get("r0", 0.337), p.get("r1", 0.251), p.get("cp", 0.651), p.get("l2d", 3.266))
        p["FINAL_OBJECT_NAME"] = self.inputs["FINAL_OBJECT_NAME"].text()
        p["type"] = self.preset_combo.currentText().split(" ")[0]
        p["INCLUDE_FINS"] = self.inputs["INCLUDE_FINS"].isChecked()

        p["balloon_params"] = {
            "ASPECT_RATIO": p.get("ASPECT_RATIO", 1.0),
            "BULGE_AMPLITUDE": p.get("BULGE_AMPLITUDE", 0.0),
            "BULGE_POWER": p.get("BULGE_POWER", 1.0),
            "GORE_AMPLITUDE": p.get("GORE_AMPLITUDE", 0.05),
            "GORE_FADE": p.get("GORE_FADE_POWER", 4.0),
            "GORE_POWER": p.get("GORE_FADE_POWER", 4.0)
        }

        is_vol = self.mode_button_group.checkedId() == 2
        hull_len = p.get("ENVELOPE_LENGTH", 100.0)

        if is_vol:
            try:
                from geometry_handler import GertlerEnvelope
                temp_env = GertlerEnvelope.from_parameters_volume(
                    p["ENVELOPE_PARAMS"],
                    self.inputs["VOLUME"].get_value(),
                    int(p["ENVELOPE_RESOLUTION"]),
                    p["LOBE_NUMBER"],
                    p["LOBE_OFFSET_X"],
                    p["LOBE_OFFSET_Y"],
                    p["LOBE_OFFSET_Z"]
                )
                hull_len = temp_env.length
                p["ENVELOPE_LENGTH"] = hull_len
            except:
                hull_len = 100.0

        req_le = (p.get("FIN_AXIAL_OFFSET", 80) / 100.0) * hull_len
        max_le = hull_len - p.get("FIN_RC_LENGTH", 15) - 0.5
        p["FIN_AXIAL_OFFSET"] = min(req_le, max_le)

        theta_text = self.inputs["FIN_THETA_POS_TEXT"].text()
        try:
            p["FIN_THETA_POS"] = [float(a.strip()) for a in theta_text.split(',') if a.strip()]
        except ValueError:
            return None

        p["OUTPUT_DIRECTORY"] = target_dir
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

    def on_worker_finished(self, result):
        matrix, stl_path = result
        self.btn_run.setEnabled(True)
        self.btn_run.setText("RUN GENERATION")

        # Immediate 3D Render
        self._update_3d_view(stl_path)

        if matrix is not None:
            for r in range(6):
                for c in range(6):
                    item = QTableWidgetItem(f"{matrix[r, c]:.4f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.matrix_table.setItem(r, c, item)
            self.log.append("[SUCCESS] Added Mass calculation complete.")
        else:
            self.matrix_table.clearContents()
            self.matrix_table.setHorizontalHeaderLabels(["u", "v", "w", "p", "q", "r"])
            self.matrix_table.setVerticalHeaderLabels(["X", "Y", "Z", "K", "M", "N"])

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
        self.load_preset(self.preset_combo.currentIndex())
        print("Parameters reset to preset defaults.")

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
