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
    QTableWidgetItem, QHeaderView, QScrollArea, QSizePolicy, QListWidget, QInputDialog
)
from pyvistaqt import BackgroundPlotter
from airfoil import get_airfoil_points, scan_airfoil_directory

# Integration of balloon.py
from balloon import create_balloon_geometry
from geometry import AirshipGeometry, plot_and_save_profile
from geometry_handler import STANDARD_ENVELOPES

# --- NEW AIRFOIL IMPORT ---
from airfoil import get_airfoil_points

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
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
                g = AirshipGeometry(self.params, self.params['SALOME_PATH'], self.params['ENVELOPE_SERIES'])

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

class LabeledSlider(QGroupBox):
    value_changed_by_user = Signal(float)

    def __init__(self, label, min_val, max_val, default_val, step, decimals=4, parent=None):
        super().__init__(label, parent)
        self.decimals, self.step = decimals, step
        self.min_val, self.max_val = min_val, max_val
        self._max_slider_val = int((max_val - min_val) / step)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        # Disable scroll behavior by ignoring wheel events
        self.slider.wheelEvent = lambda event: event.ignore()

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
        except:
            pass

    def get_value(self):
        return float(self.value_editor.text())

    def set_value(self, val):
        slider_pos = int((val - self.min_val) / self.step)
        self.slider.setValue(slider_pos)
        self.value_editor.setText(f"{val:.{self.decimals}f}")

class AirshipGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STRATOS: Software-based Tool for Rapid Airship Translation & Optimized Shaping")

        # --- NEW: AUTO-SET ASPECT RATIO & FULLSCREEN ---
        # 1. Get the primary screen information
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # 2. Set a fallback size (90% of screen) if user restores from maximized
        width = int(screen_width * 0.9)
        height = int(screen_height * 0.9)
        self.resize(width, height)

        # 3. Default to Maximized (Taskbar visible)
        # This replaces the fixed setGeometry(100, 100, 1400, 950)
        self.showMaximized()
        # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # --- EXISTING INITIALIZATION ---
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
        # Add stretch to make the tab widget take up all available vertical space
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.primary_input_tab = QWidget()
        self.fairings_tab = QWidget()
        self.fin_tab = QWidget()
        self.wing_tab = QWidget()
        self.airfoil_tab = QWidget() # --- NEW AIRFOIL TAB INSTANCE ---
        self.output_tab = QWidget()

        self.setup_primary_tab_layout()
        self.setup_aerostat_tab()
        self.setup_fairings_tab()
        self.setup_fin_tab()
        self.setup_wing_tab()
        self.setup_airfoil_tab()     # --- INITIALIZE AIRFOIL TAB ---
        self.setup_output_tab()

        self.main_layout.addWidget(self.tab_widget)

        # Navigation buttons (Next/Prev/Exit/Reset)
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
        for i, name in enumerate(["STANDARD MODE", "VOLUMETRIC MODE", "SUPER PRESSURE BALLOON", "AEROSTAT"], 1):
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
            QTextEdit { font-family: Cascadia Code; }
            QHeaderView::section { background-color: #2D2D2D; color: #00BFFF; padding: 4px; border: 1px solid #3c3c3c; font-weight: bold; }
            QComboBox { background-color: #3C3C3C; border: 1px solid #00BFFF; padding: 5px; border-radius: 4px; color: #FFFFFF; }
            QComboBox#LargeDropdown { border: 2px solid #00BFFF; padding: 8px; border-radius: 4px; font-size: 11pt; font-weight: bold; }
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

        # FIX: Force equal column width so hiding widgets doesn't resize columns
        cl.setColumnStretch(0, 1)
        cl.setColumnStretch(1, 1)

        # --- NEW: Envelope Series Selector ---
        self.inputs["ENVELOPE_SERIES"] = QComboBox()
        self.inputs["ENVELOPE_SERIES"].setObjectName("LargeDropdown")
        self.inputs["ENVELOPE_SERIES"].addItems(["GERTLER", "NACA", "DRAGON_DREAM"])
        self.inputs["ENVELOPE_SERIES"].setMinimumHeight(45) # Maintained large height
        self.inputs["ENVELOPE_SERIES"].currentIndexChanged.connect(self._update_series_visibility)

        cl.addWidget(QLabel("Profile Series:"), 0, 0)
        cl.addWidget(self.inputs["ENVELOPE_SERIES"], 0, 1, 1, 2) # Maintained 2-column span

        # --- Existing Presets ---
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("LargeDropdown")
        self.preset_combo.setMinimumHeight(45)
        self.preset_combo.addItems(list(STANDARD_ENVELOPES.keys()))
        self.preset_combo.currentIndexChanged.connect(self.load_preset)

        self.preset_label = QLabel("Shape Preset:")
        cl.addWidget(self.preset_label, 1, 0)
        cl.addWidget(self.preset_combo, 1, 1, 1, 2)

        # --- Parameters ---
        self.inputs["l2d"] = LabeledSlider("Length/Diameter Ratio (L/D)", 1, 8, 3.266, 0.0001, 4)
        self.inputs["m1"] = LabeledSlider("Max. Thickness Position (m1)", 0.3, 0.6, 0.419, 0.0001, 4)
        self.inputs["r0"] = LabeledSlider("Nose Radius (r0)", 0.01, 1, 0.337, 0.0001, 4)
        self.inputs["r1"] = LabeledSlider("Stern Radius (r1)", 0.01, 1, 0.251, 0.0001, 4)
        self.inputs["cp"] = LabeledSlider("Prismatic Coefficient (cp)", 0.5, 0.8, 0.651, 0.0001, 4)
        self.inputs["ENVELOPE_RESOLUTION"] = LabeledSlider("Resolution", 50, 500, 150, 1, 0)
        self.inputs["HULL_WIDTH"] = LabeledSlider("Max Width (W)", 5.0, 100.0, 29.5, 0.1, 2)
        self.inputs["HULL_HEIGHT"] = LabeledSlider("Max Height (H)", 5.0, 100.0, 14.0, 0.1, 2)
        self.inputs["BOTTOM_FLATNESS"] = LabeledSlider("Bottom Flatness %", 0.0, 100.0, 25.0, 1.0, 0)

        # Add them to the layout (cl)
        cl.addWidget(self.inputs["HULL_WIDTH"], 5, 0)
        cl.addWidget(self.inputs["HULL_HEIGHT"], 5, 1)
        cl.addWidget(self.inputs["BOTTOM_FLATNESS"], 6, 0)

        # REORDERED WIDGETS
        # Row 2: L/D and m1
        cl.addWidget(self.inputs["l2d"], 2, 0)
        cl.addWidget(self.inputs["m1"], 2, 1)

        # Row 3: Resolution (Below L/D) and r0
        cl.addWidget(self.inputs["ENVELOPE_RESOLUTION"], 3, 0)
        cl.addWidget(self.inputs["r0"], 3, 1)

        # Row 4: r1 and cp
        cl.addWidget(self.inputs["r1"], 4, 0)
        cl.addWidget(self.inputs["cp"], 4, 1)

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

        self.length_box = QGroupBox("Standard Mode: Length (m)")
        ll = QVBoxLayout(self.length_box)
        self.inputs["ENVELOPE_LENGTH"] = LabeledSlider("Length (L)", 0, 500, 100, 0.0001, 4)
        ll.addWidget(self.inputs["ENVELOPE_LENGTH"])
        layout.addWidget(self.length_box)

        self.volume_box = QGroupBox("Volumetric Mode: Volume")
        vl = QVBoxLayout(self.volume_box)
        self.inputs["VOLUME"] = LabeledSlider("Volume (m³)", 0, 1000000, 5000, 0.1, 4)
        vl.addWidget(self.inputs["VOLUME"])
        layout.addWidget(self.volume_box)

        # --- NEW: Appendages Box for Wing Toggle ---
        self.appendages_box = QGroupBox("Hull Appendages")
        al = QVBoxLayout(self.appendages_box)

        self.inputs["INCLUDE_WINGS"] = QCheckBox("GENERATE WING WITH HULL")
        self.inputs["INCLUDE_WINGS"].setChecked(False) # Default to False so the tab is hidden initially
        self.inputs["INCLUDE_WINGS"].setFont(QFont("Arial", 10, QFont.Bold))
        self.inputs["INCLUDE_WINGS"].setStyleSheet("color: #00BFFF;")

        # Connect to refresh_tabs to dynamically add/remove the wing tab
        self.inputs["INCLUDE_WINGS"].toggled.connect(self.refresh_tabs)
        self.inputs["INCLUDE_WINGS"].toggled.connect(self._auto_update_props)

        al.addWidget(self.inputs["INCLUDE_WINGS"])
        layout.addWidget(self.appendages_box)

        layout.addStretch()

    def setup_wing_tab(self):
        main_layout = QVBoxLayout(self.wing_tab)
        main_layout.setContentsMargins(15, 15, 15, 15)

        wing_dim_group = QGroupBox("Wing Placement & Defaults")
        w_layout = QGridLayout(wing_dim_group)
        self.inputs["WING_AXIAL_OFFSET"] = LabeledSlider("Root Axial Pos (m)", 0.0, 200.0, 40.0, 0.1, 2)
        self.inputs["WING_THICKNESS"] = LabeledSlider("Default Thickness (%)", 5.0, 25.0, 12.0, 0.1, 2)
        w_layout.addWidget(self.inputs["WING_AXIAL_OFFSET"], 0, 0)
        w_layout.addWidget(self.inputs["WING_THICKNESS"], 0, 1)
        main_layout.addWidget(wing_dim_group)

        # Dynamic Stations Table
        station_group = QGroupBox("Wing Stations (Root to Tip)")
        s_layout = QVBoxLayout(station_group)

        self.wing_table = QTableWidget(0, 6) # Reverted back to 6 columns
        self.wing_table.setHorizontalHeaderLabels(["Y (Span)", "Chord", "Twist (°)", "Sweep Offset (X)", "Dihedral Offset (Z)", "Airfoil"])
        self.wing_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.wing_table.setStyleSheet("QTableWidget { background-color: #2D2D2D; color: #FFFFFF; }")

        btn_layout = QHBoxLayout()
        self.btn_add_station = QPushButton("+ Add Station")
        self.btn_add_station.setStyleSheet("background-color: #3C3C3C; color: #00BFFF; font-weight: bold;")
        self.btn_rem_station = QPushButton("- Remove Station")

        self.btn_add_station.clicked.connect(lambda: self._add_wing_station())
        self.btn_rem_station.clicked.connect(self._remove_wing_station)

        btn_layout.addWidget(self.btn_add_station)
        btn_layout.addWidget(self.btn_rem_station)

        s_layout.addLayout(btn_layout)
        s_layout.addWidget(self.wing_table)
        main_layout.addWidget(station_group, 1)

        # Initialize with standard Root and Tip
        self._add_wing_station(0.0, 5.0, 2.0, 0.0, 0.0)
        self._add_wing_station(10.0, 2.0, -2.0, 2.58, 0.87)

    def _add_wing_station(self, y=0.0, c=1.0, t=0.0, x=0.0, z=0.0):
        row = self.wing_table.rowCount()
        self.wing_table.insertRow(row)
        self.wing_table.setItem(row, 0, QTableWidgetItem(f"{y:.2f}"))
        self.wing_table.setItem(row, 1, QTableWidgetItem(f"{c:.2f}"))
        self.wing_table.setItem(row, 2, QTableWidgetItem(f"{t:.2f}"))
        self.wing_table.setItem(row, 3, QTableWidgetItem(f"{x:.2f}"))
        self.wing_table.setItem(row, 4, QTableWidgetItem(f"{z:.2f}"))

        # Airfoil Selector (Pulls directly from the Airfoil Workspace)
        combo = QComboBox()
        if hasattr(self, 'airfoil_library'):
            combo.addItems(list(self.airfoil_library.keys()))
        self.wing_table.setCellWidget(row, 5, combo)

    def _sync_wing_dropdowns(self):
        """Called whenever the Airfoil Library changes to update the Wing table without wiping selections."""
        if not hasattr(self, 'wing_table'): return
        for row in range(self.wing_table.rowCount()):
            combo = self.wing_table.cellWidget(row, 5)
            if combo:
                current = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                if hasattr(self, 'airfoil_library'):
                    combo.addItems(list(self.airfoil_library.keys()))
                    if current in self.airfoil_library:
                        combo.setCurrentText(current)
                combo.blockSignals(False)

    def _remove_wing_station(self):
        row = self.wing_table.rowCount()
        if row > 2:  # Prevent removing the last 2 stations (need at least root and tip)
            self.wing_table.removeRow(row - 1)

    def setup_airfoil_tab(self):
        # Initialize the Central Asset Library
        self.airfoil_library = {}

        main_layout = QVBoxLayout(self.airfoil_tab)
        self.airfoil_splitter = QSplitter(Qt.Horizontal)

        # --- LEFT PANE (The Asset Manager Workspace) ---
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)

        self.airfoil_list = QListWidget()
        self.airfoil_list.setStyleSheet("background-color: #2D2D2D; color: #00BFFF; font-weight: bold; font-size: 11pt;")
        self.airfoil_list.currentItemChanged.connect(self._on_airfoil_selected)

        btn_layout = QHBoxLayout()
        self.btn_import_files = QPushButton("Import .dat Files")
        self.btn_import_files.setStyleSheet("background-color: #3C3C3C; font-weight: bold;")
        self.btn_import_files.clicked.connect(self._import_airfoil_files)

        self.btn_add_naca = QPushButton("+ Add NACA")
        self.btn_add_naca.clicked.connect(self._add_naca_profile)

        btn_layout.addWidget(self.btn_import_files)
        btn_layout.addWidget(self.btn_add_naca)

        self.btn_remove_airfoil = QPushButton("- Remove Selected")
        self.btn_remove_airfoil.setStyleSheet("color: #FF6347;")
        self.btn_remove_airfoil.clicked.connect(self._remove_airfoil)

        left_layout.addWidget(QLabel("Imported Airfoil Library:"))
        left_layout.addWidget(self.airfoil_list)
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.btn_remove_airfoil)

        # --- RIGHT PANE (Editor & Preview) ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)

        self.af_editor_group = QGroupBox("Selected Airfoil Properties")
        editor_layout = QGridLayout(self.af_editor_group)

        # Dynamic Inputs
        self.inputs["AF_METHOD"] = QComboBox()
        self.inputs["AF_METHOD"].addItems(["Auto (Best Fit)", "cst", "parsec"])
        self.inputs["AF_THICKNESS"] = LabeledSlider("NACA Thickness (%)", 1.0, 50.0, 12.0, 0.1, 1)
        self.inputs["AF_RES"] = LabeledSlider("Resolution", 10, 500, 100, 1, 0)
        self.inputs["AF_SCALE"] = LabeledSlider("Scale Factor", 0.01, 10.0, 1.0, 0.01, 2)
        self.inputs["AF_TX"] = LabeledSlider("Translation X", -50.0, 50.0, 0.0, 0.01, 2)
        self.inputs["AF_TY"] = LabeledSlider("Translation Y", -50.0, 50.0, 0.0, 0.01, 2)
        self.inputs["AF_ROT"] = LabeledSlider("Rotation Angle (°)", -180.0, 180.0, 0.0, 0.1, 1)

        # Connect all changes to the sync function
        self.inputs["AF_METHOD"].currentIndexChanged.connect(self._sync_airfoil_props)
        for key in ["AF_THICKNESS", "AF_RES", "AF_SCALE", "AF_TX", "AF_TY", "AF_ROT"]:
            self.inputs[key].value_changed_by_user.connect(self._sync_airfoil_props)

        self.method_label = QLabel("Fitting Method:")
        editor_layout.addWidget(self.method_label, 0, 0)
        editor_layout.addWidget(self.inputs["AF_METHOD"], 0, 1)
        editor_layout.addWidget(self.inputs["AF_THICKNESS"], 1, 0, 1, 2)
        editor_layout.addWidget(self.inputs["AF_RES"], 2, 0)
        editor_layout.addWidget(self.inputs["AF_SCALE"], 2, 1)
        editor_layout.addWidget(self.inputs["AF_TX"], 3, 0)
        editor_layout.addWidget(self.inputs["AF_TY"], 3, 1)
        editor_layout.addWidget(self.inputs["AF_ROT"], 4, 0, 1, 2)

        right_layout.addWidget(self.af_editor_group)

        # Plot
        plot_group = QGroupBox("Geometry Preview")
        plot_layout = QVBoxLayout(plot_group)
        self.airfoil_fig = Figure(facecolor='#1e1e1e', tight_layout=True)
        self.airfoil_canvas = FigureCanvas(self.airfoil_fig)
        plot_layout.addWidget(self.airfoil_canvas)
        right_layout.addWidget(plot_group, 1)

        self.airfoil_splitter.addWidget(left_pane)
        self.airfoil_splitter.addWidget(right_pane)
        self.airfoil_splitter.setStretchFactor(0, 1)
        self.airfoil_splitter.setStretchFactor(1, 3)

        main_layout.addWidget(self.airfoil_splitter)

        default_name = "NACA_0012"
        self.airfoil_library[default_name] = {
            "type": "NACA", "thickness": 12.0,
            "res": 100, "scale": 1.0, "tx": 0.0, "ty": 0.0, "rot": 0.0
        }
        self.airfoil_list.addItem(default_name)
        self.airfoil_list.setCurrentRow(0)

        self._sync_wing_dropdowns()
        self._update_editor_visibility()

    def _import_airfoil_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Airfoils", "", "Data Files (*.dat *.txt *.csv)")
        for filepath in files:
            if os.path.getsize(filepath) == 0: continue

            # Simple validation
            valid_lines = sum(1 for line in open(filepath, 'r') if len(line.strip().replace(',', ' ').split()) >= 2)
            if valid_lines < 10: continue

            base_name = os.path.splitext(os.path.basename(filepath))[0]
            name = base_name
            count = 1
            while name in self.airfoil_library:
                name = f"{base_name}_{count}"
                count += 1

            self.airfoil_library[name] = {
                "type": "FILE", "path": filepath, "method": "Auto (Best Fit)",
                "res": 100, "scale": 1.0, "tx": 0.0, "ty": 0.0, "rot": 0.0
            }
            self.airfoil_list.addItem(name)

        self.airfoil_list.setCurrentRow(self.airfoil_list.count() - 1)
        self._sync_wing_dropdowns()

    def _add_naca_profile(self):
        # 1. Prompt the user for the NACA 4-digit number
        text, ok = QInputDialog.getText(self, "Add NACA Airfoil", "Enter NACA 4-digit symmetric series (e.g., 0012, 0015, 0021):")

        if ok and text.strip():
            naca_str = text.strip()

            # Basic validation: ensure it's a number and has at least 2 digits to extract thickness
            if not naca_str.isdigit() or len(naca_str) < 2:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid NACA number (e.g., 0012).")
                return

            # 2. Extract thickness from the last two digits (e.g., "0015" -> 15.0)
            thickness = float(naca_str[-2:])

            # Enforce symmetric limitation based on STRATOS's native NACA capabilities
            if len(naca_str) == 4 and not naca_str.startswith("00"):
                self.log.append(f"[WARNING] STRATOS native generator supports symmetric profiles. Defaulting to NACA 00{int(thickness):02d}.")
                naca_str = f"00{int(thickness):02d}"

            # 3. Handle naming and duplicates
            name = f"NACA_{naca_str}"
            count = 1
            base_name = name
            while name in self.airfoil_library:
                name = f"{base_name}_{count}"
                count += 1

            # 4. Add to the Asset Manager library
            self.airfoil_library[name] = {
                "type": "NACA", "thickness": thickness,
                "res": 100, "scale": 1.0, "tx": 0.0, "ty": 0.0, "rot": 0.0
            }

            # 5. Add to UI list and trigger the auto-plot
            self.airfoil_list.addItem(name)

            # Setting the current row automatically triggers _on_airfoil_selected(),
            # which in turn calls _generate_and_plot_airfoil() to show the plot!
            self.airfoil_list.setCurrentRow(self.airfoil_list.count() - 1)
            self._sync_wing_dropdowns()

    def _remove_airfoil(self):
        item = self.airfoil_list.currentItem()
        if item and len(self.airfoil_library) > 1: # Prevent deleting the last airfoil
            name = item.text()
            del self.airfoil_library[name]
            self.airfoil_list.takeItem(self.airfoil_list.row(item))
            self._sync_wing_dropdowns()
        elif len(self.airfoil_library) <= 1:
            QMessageBox.warning(self, "Warning", "You must have at least one airfoil in the library.")

    def _on_airfoil_selected(self, current, previous):
        self._update_editor_visibility()
        if not current: return
        name = current.text()
        data = self.airfoil_library[name]

        # Block signals to update sliders without triggering recursion
        self._block_af_signals(True)
        if data["type"] == "FILE":
            self.inputs["AF_METHOD"].setCurrentText(data["method"])
        else:
            self.inputs["AF_THICKNESS"].set_value(data["thickness"])

        self.inputs["AF_RES"].set_value(data["res"])
        self.inputs["AF_SCALE"].set_value(data["scale"])
        self.inputs["AF_TX"].set_value(data["tx"])
        self.inputs["AF_TY"].set_value(data["ty"])
        self.inputs["AF_ROT"].set_value(data["rot"])
        self._block_af_signals(False)
        self._generate_and_plot_airfoil()

    def _block_af_signals(self, block):
        for key in ["AF_METHOD", "AF_THICKNESS", "AF_RES", "AF_SCALE", "AF_TX", "AF_TY", "AF_ROT"]:
            self.inputs[key].blockSignals(block)

    def _update_editor_visibility(self):
        item = self.airfoil_list.currentItem()
        if not item:
            self.af_editor_group.setEnabled(False)
            return
        self.af_editor_group.setEnabled(True)
        is_file = self.airfoil_library[item.text()]["type"] == "FILE"
        self.inputs["AF_METHOD"].setVisible(is_file)
        self.method_label.setVisible(is_file)
        self.inputs["AF_THICKNESS"].setVisible(not is_file)

    def _sync_airfoil_props(self, *args):
        item = self.airfoil_list.currentItem()
        if not item: return
        data = self.airfoil_library[item.text()]

        if data["type"] == "FILE":
            data["method"] = self.inputs["AF_METHOD"].currentText()
        else:
            data["thickness"] = self.inputs["AF_THICKNESS"].get_value()

        data["res"] = int(self.inputs["AF_RES"].get_value())
        data["scale"] = self.inputs["AF_SCALE"].get_value()
        data["tx"] = self.inputs["AF_TX"].get_value()
        data["ty"] = self.inputs["AF_TY"].get_value()
        data["rot"] = self.inputs["AF_ROT"].get_value()

        self._generate_and_plot_airfoil()

    def _generate_and_plot_airfoil(self):
        item = self.airfoil_list.currentItem()
        if not item: return
        data = self.airfoil_library[item.text()]

        try:
            res, scale = data["res"], data["scale"]
            trans, rot = (data["tx"], data["ty"]), data["rot"]

            if data["type"] == "FILE":
                meth = None if "Auto" in data["method"] else data["method"]
                x, y = get_airfoil_points(filename=data["path"], method=meth, resolution=res, scale_factor=scale, translation=trans, rotation_angle=rot)
            else:
                x, y = get_airfoil_points(thickness=data["thickness"], resolution=res, scale_factor=scale, translation=trans, rotation_angle=rot)

            # Plot
            self.airfoil_fig.clear()
            ax = self.airfoil_fig.add_subplot(111)
            ax.plot(x, y, color='#00BFFF', linewidth=2, marker='o', markersize=3)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_facecolor('#1e1e1e')
            for spine in ax.spines.values(): spine.set_color('#3c3c3c')
            ax.tick_params(colors='white')
            self.airfoil_fig.tight_layout()
            self.airfoil_canvas.draw()
        except Exception as e:
            self.log.append(f"[ERROR] Plot failed: {e}")

    def _update_series_visibility(self):
        """Toggles visibility of inputs based on selected Envelope Series (Gertler/NACA/Dragon Dream)."""
        series = self.inputs["ENVELOPE_SERIES"].currentText()
        is_naca = series == "NACA"
        is_dragon = series == "DRAGON_DREAM"

        gertler_widgets = ["m1", "r0", "r1", "cp", "l2d"]
        dragon_widgets = ["HULL_WIDTH", "HULL_HEIGHT", "BOTTOM_FLATNESS"]

        for k in gertler_widgets:
            if k in self.inputs: self.inputs[k].setVisible(not is_naca and not is_dragon)

        for k in dragon_widgets:
            if k in self.inputs: self.inputs[k].setVisible(is_dragon)

        self.preset_combo.setVisible(not is_naca and not is_dragon)
        self.preset_label.setVisible(not is_naca and not is_dragon)

        # --- NEW: Disable the 2D Petal Plot button for Dragon Dream ---
        if hasattr(self, 'btn_plot'):
            self.btn_plot.setEnabled(not is_dragon)

    def setup_aerostat_tab(self):
        self.aerostat_tab = QWidget()
        tab_layout = QVBoxLayout(self.aerostat_tab)

        # Create the Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")

        # Container widget for all the content
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(content_widget)

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
        self.inputs["DELTA_P"] = LabeledSlider("Delta P (Pa)", 0, 1000, 500, 1, 0)
        self.inputs["DELTA_T"] = LabeledSlider("Delta T (K)", 0, 20, 5, 0.1, 1)
        gl.addWidget(self.inputs["GAS_PURITY"], 0, 0)
        gl.addWidget(self.inputs["GAS_CONSTANT"], 0, 1)
        gl.addWidget(self.inputs["DELTA_P"], 1, 0)
        gl.addWidget(self.inputs["DELTA_T"], 1, 1)
        layout.addWidget(gas_grp)

        # 3. Ballonet Configuration
        ballonet_grp = QGroupBox("Ballonet Configuration")
        bl = QGridLayout(ballonet_grp)
        self.inputs["BALLONET_NUMBER"] = LabeledSlider("Number of Ballonets", 0, 4, 2, 1, 0)
        self.inputs["BALLONET_SHAPE"] = QComboBox()
        self.inputs["BALLONET_SHAPE"].addItems(["THREE_QUARTER", "HEMISPHERE"])
        self.inputs["BALLONET_FABRIC_DENSITY"] = LabeledSlider("Fabric Density (kg/m²)", 0.1, 1.0, 0.35, 0.01, 2)
        bl.addWidget(QLabel("Ballonet Shape:"), 0, 0)
        bl.addWidget(self.inputs["BALLONET_SHAPE"], 0, 1)
        bl.addWidget(self.inputs["BALLONET_NUMBER"], 1, 0)
        bl.addWidget(self.inputs["BALLONET_FABRIC_DENSITY"], 1, 1)
        layout.addWidget(ballonet_grp)

        # 4. Mass and Structural Design
        mass_grp = QGroupBox("Mass and Structural Design")
        ml = QGridLayout(mass_grp)
        self.inputs["SKIN_DENSITY"] = LabeledSlider("Skin Density (kg/m²)", 0.1, 2.0, 0.75, 0.01, 2)
        self.inputs["PAYLOAD_MASS"] = LabeledSlider("Payload/Add. Mass (kg)", 0, 5000, 220, 1, 1)
        self.inputs["SKIN_THICKNESS"] = LabeledSlider("Skin Thick. (mm)", 0.1, 5.0, 1.0, 0.1, 2)

        # Tether Mass Controls
        self.inputs["INCLUDE_TETHER"] = QCheckBox("INCLUDE TETHER MASS")
        self.inputs["INCLUDE_TETHER"].setChecked(True)
        self.inputs["INCLUDE_TETHER"].setStyleSheet("color: #00BFFF; font-weight: bold;")
        self.inputs["INCLUDE_TETHER"].toggled.connect(self._toggle_tether_inputs)

        self.inputs["TETHER_DENSITY"] = LabeledSlider("Tether Density (kg/m)", 0, 5, 0.1, 0.01, 2)
        self.inputs["TETHER_FRACTION"] = LabeledSlider("Tether Fraction (0-1)", 0, 1, 1.0, 0.01, 2)
        self.inputs["TARGET_NET_LIFT"] = LabeledSlider("Target Net Lift (N)", -1000, 5000, 0, 1, 1)

        self.inputs["OPTIMIZE_LENGTH"] = QCheckBox("OPTIMIZE LENGTH FOR TARGET LIFT")
        self.inputs["OPTIMIZE_LENGTH"].setChecked(True)
        self.inputs["OPTIMIZE_LENGTH"].setFont(QFont("Arial", 10, QFont.Bold))

        ml.addWidget(self.inputs["SKIN_DENSITY"], 0, 0)
        ml.addWidget(self.inputs["PAYLOAD_MASS"], 0, 1)
        ml.addWidget(self.inputs["INCLUDE_TETHER"], 1, 0)
        ml.addWidget(self.inputs["SKIN_THICKNESS"], 1, 1)
        ml.addWidget(self.inputs["TETHER_DENSITY"], 2, 0)
        ml.addWidget(self.inputs["TETHER_FRACTION"], 2, 1)
        ml.addWidget(self.inputs["TARGET_NET_LIFT"], 3, 0)
        ml.addWidget(self.inputs["OPTIMIZE_LENGTH"], 4, 0, 1, 2)
        layout.addWidget(mass_grp)

        # 5. Thermal & Stress Analysis Inputs
        thermal_grp = QGroupBox("Thermal & Stress Analysis")
        tl = QGridLayout(thermal_grp)
        self.inputs["MATERIAL_CLASS"] = QComboBox()
        self.inputs["MATERIAL_CLASS"].addItems(["Standard", "High temperature", "Cold temperature", "Extreme environment"])

        self.inputs["SAFETY_FACTOR"] = LabeledSlider("Safety Factor", 1.0, 10.0, 4.0, 0.1, 1)
        self.inputs["SOLAR_FLUX"] = LabeledSlider("Solar Flux (W/m²)", 0, 2000, 1000, 10, 0)
        self.inputs["WIND_SPEED"] = LabeledSlider("Wind Speed (m/s)", 0, 50, 5, 1, 0)
        self.inputs["EMISSIVITY"] = LabeledSlider("Emissivity", 0.0, 1.0, 0.8, 0.01, 2)
        self.inputs["ABSORPTIVITY"] = LabeledSlider("Absorptivity", 0.0, 1.0, 0.3, 0.01, 2)

        # --- NEW UI ELEMENTS ---
        self.inputs["FATIGUE_FACTOR"] = LabeledSlider("Fatigue Factor/Yr", 0.8, 1.0, 0.995, 0.001, 3)
        self.inputs["UV_DEGRADATION"] = LabeledSlider("UV Degrade/Yr", 0.0, 0.2, 0.02, 0.001, 3)

        tl.addWidget(QLabel("Envelope Material:"), 0, 0)
        tl.addWidget(self.inputs["MATERIAL_CLASS"], 0, 1)
        tl.addWidget(self.inputs["SAFETY_FACTOR"], 1, 0)
        tl.addWidget(self.inputs["SOLAR_FLUX"], 1, 1)
        tl.addWidget(self.inputs["WIND_SPEED"], 2, 0)
        tl.addWidget(self.inputs["EMISSIVITY"], 2, 1)
        tl.addWidget(self.inputs["ABSORPTIVITY"], 3, 0)
        tl.addWidget(self.inputs["FATIGUE_FACTOR"], 3, 1) # Added to grid
        tl.addWidget(self.inputs["UV_DEGRADATION"], 4, 0) # Added to grid
        layout.addWidget(thermal_grp)

        layout.addStretch()

        # Finalize the Scroll Area
        scroll.setWidget(content_widget)
        tab_layout.addWidget(scroll)

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

        # --- Check if wings are enabled ---
        is_winged = self.inputs.get("INCLUDE_WINGS") and self.inputs["INCLUDE_WINGS"].isChecked() and not is_balloon

        # Toggle Input group visibility
        self.hull_shape_box.setHidden(is_balloon)
        self.balloon_params_box.setHidden(not is_balloon)
        self.hull_config_group.setHidden(is_balloon)
        self.length_box.setHidden(not is_standard)
        self.volume_box.setHidden(not (is_vol or is_balloon))
        self.appendages_box.setHidden(is_balloon) # Hide the wing toggle in balloon mode

        # --- FIX: Toggle Output Tab Panels based on Mode ---
        if hasattr(self, 'aero_analysis_group'):
            # Show graphs and CSV button only in Aerostat Mode
            self.aero_analysis_group.setVisible(is_aero)
            self.btn_csv.setVisible(is_aero)

            # Hide 3D Viewer, Added Mass Matrix, and 2D Plot button in Aerostat Mode
            self.preview_group.setVisible(not is_aero)
            self.matrix_group.setVisible(not is_aero)
            self.btn_plot.setVisible(not is_aero)
        # ---------------------------------------------------

        # Update available tabs
        curr = self.tab_widget.currentIndex()
        self.tab_widget.clear()
        self.tab_widget.addTab(self.primary_input_tab, "Envelope Geometry")

        if is_aero:
            self.tab_widget.addTab(self.aerostat_tab, "Aerostat Analysis")
        if is_multi:
            self.tab_widget.addTab(self.fairings_tab, "Multi-Lobe Configuration")
        if is_winged:
            self.tab_widget.addTab(self.airfoil_tab, "Airfoil Design")
            self.tab_widget.addTab(self.wing_tab, "Wing Design")
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
        self.inputs["FIN_DENSITY"] = LabeledSlider("Fin Density (kg/m³)", 1.0, 50.0, 10.0, 0.1, 1)
        self.inputs["FIN_NUMBER"] = LabeledSlider("N Fins", 2, 8, 4, 1, decimals=0)

        fin_sweep_layout.addWidget(self.inputs["FIN_SWEEP_ANGLE"], 0, 0)
        fin_sweep_layout.addWidget(self.inputs["FIN_TIP_ANGLE"], 0, 1)
        fin_sweep_layout.addWidget(self.inputs["FIN_DENSITY"], 0, 2)
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
        # UPDATED: Added CV to labels and keys
        labels = ["Vol (m³):", "Surf (m²):", "Top (m²):", "Side (m²):", "CV (x, y):"]
        keys = ["vol", "surf", "top_area", "side_area", "cv"]

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

        self.aero_analysis_group = QGroupBox("Aerostat Performance Analysis")
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

        # --- UPDATED SIGNAL CONNECTIONS FOR AUTO-UPDATE ---
        geo_keys = [
            "l2d", "m1", "r0", "r1", "cp", "ENVELOPE_LENGTH", "VOLUME",
            "LOBE_OFFSET_X_SLIDER", "LOBE_OFFSET_Y_SLIDER", "LOBE_OFFSET_Z_SLIDER",
            "FIN_RC_LENGTH", "FIN_HEIGHT", "FIN_THICKNESS", "FIN_TAPER_RATIO",
            "FIN_NUMBER",
            # --- NEW: ALL Wing Keys Included ---
            "WING_SPAN", "WING_ROOT_CHORD", "WING_TIP_CHORD", "WING_THICKNESS",
            "WING_SWEEP", "WING_DIHEDRAL", "WING_TWIST_ROOT", "WING_TWIST_TIP"
        ]

        for key in geo_keys:
            if key in self.inputs:
                if hasattr(self.inputs[key], 'value_changed_by_user'):
                    self.inputs[key].value_changed_by_user.connect(self._auto_update_props)

        self.inputs["INCLUDE_FINS"].toggled.connect(self._auto_update_props)
        # --- NEW: Connect Wing Toggle ---
        if "INCLUDE_WINGS" in self.inputs:
            self.inputs["INCLUDE_WINGS"].toggled.connect(self._auto_update_props)

        self.inputs["ENVELOPE_SERIES"].currentIndexChanged.connect(self._auto_update_props)
        self.preset_combo.currentIndexChanged.connect(self._auto_update_props)

    def handle_output_action(self):
        """Routes the button click based on mode; skips generation in Aero mode."""
        if self.mode_button_group.checkedId() == 4:
            self.run_instant_aerostat()
        else:
            self.run_process()

    def run_instant_aerostat(self):
        """
        Performs analytical performance calculations by initializing the AerostatHull
        once and handling optimization and property retrieval in a single pass.
        """
        self.log.append("\n[PROCESS] Running Analytical Aerostat Solver...")
        try:
            target_dir = self.current_session_folder
            p = self.get_parameters(target_dir)

            from aerostat import AerostatHull, get_atmospheric_properties, get_thermal_model, get_gas_mass
            from geometry_handler import GertlerEnvelope, NACAEnvelope

            series = p.get("ENVELOPE_SERIES", "GERTLER")
            if series == "NACA":
                resolved_env = NACAEnvelope.from_parameters((p["l2d"],), p["ENVELOPE_LENGTH"])
            else:
                resolved_env = GertlerEnvelope.from_parameters(p["ENVELOPE_PARAMS"], p["ENVELOPE_LENGTH"])

            # Map UI Material dropdown to physical properties natively
            MATERIAL_PROPS = {
                "Standard": {"cte": 2.3e-5, "base_strength": 75.0, "temp_derating": 0.15},
                "High temperature": {"cte": 1.5e-5, "base_strength": 90.0, "temp_derating": 0.05},
                "Cold temperature": {"cte": 3.0e-5, "base_strength": 60.0, "temp_derating": 0.20},
                "Extreme environment": {"cte": 1.0e-5, "base_strength": 120.0, "temp_derating": 0.01}
            }
            mat = MATERIAL_PROPS[p["MATERIAL_CLASS"]]

            ahull = AerostatHull(
                envelope=resolved_env,
                skin_density=p["SKIN_DENSITY"],
                skin_thickness=p.get("SKIN_THICKNESS", 1.0) / 1000.0, # Convert mm to m
                additional_mass=p["PAYLOAD_MASS"],
                operational_height=p["OPERATIONAL_HEIGHT"],
                deployment_height=0,
                margin_height=p["MARGIN_HEIGHT"],
                RH=p["RELATIVE_HUMIDITY"],
                purity=p["GAS_PURITY"],
                delta_P=p["DELTA_P"],
                delta_T=p["DELTA_T"],
                gas_constant=p["GAS_CONSTANT"],
                lobe_number=p["LOBE_NUMBER"],
                e=p["LOBE_OFFSET_X"], f=p["LOBE_OFFSET_Y"], g=p["LOBE_OFFSET_Z"],
                ballonet_number=int(p["BALLONET_NUMBER"]),
                ballonet_shape=p["BALLONET_SHAPE"],
                ballonet_fabric_density=p["BALLONET_FABRIC_DENSITY"],
                tether_density=p["TETHER_DENSITY"],
                tether_fraction=p["TETHER_FRACTION"],
                fin_rc=p.get("FIN_RC_LENGTH", 0),
                fin_height=p.get("FIN_HEIGHT", 0),
                fin_taper_ratio=p.get("FIN_TAPER_RATIO", 1),
                fin_thickness=p.get("FIN_THICKNESS", 0),
                fin_number=p.get("FIN_NUMBER", 4),
                fin_density=p.get("FIN_DENSITY", 10.0),

                # --- NEW: Wing Parameters ---
                has_wings=p.get("INCLUDE_WINGS", False),
                # Dynamically extract span and chords from the first and last stations in the table
                wing_span=(p.get("WING_STATIONS", [{}])[-1].get('y', 0) * 2) if p.get("WING_STATIONS") else 0,
                wing_root_chord=p.get("WING_STATIONS", [{}])[0].get('chord', 0),
                wing_tip_chord=p.get("WING_STATIONS", [{}])[-1].get('chord', 0),
                wing_thickness=p.get("WING_THICKNESS", 0),
                wing_density=p.get("FIN_DENSITY", 10.0), # Reuses fin density for structural consistency
                # ----------------------------

                cte=mat["cte"],
                base_strength=mat["base_strength"],
                temp_derating=mat["temp_derating"],
                solar_flux=p["SOLAR_FLUX"],
                emissivity=p["EMISSIVITY"],
                absorptivity=p["ABSORPTIVITY"],
                wind_speed=p["WIND_SPEED"]
            )

            if self.inputs.get("OPTIMIZE_LENGTH") and self.inputs["OPTIMIZE_LENGTH"].isChecked():
                target_lift = p.get("TARGET_NET_LIFT", 0)
                optimized_env, convergence_error = ahull.initialise_from_operational_altitude([1.0, 1e10], target_lift=target_lift)
                self.log.append(f"[INFO] Optimized Length: {optimized_env.length:.3f} m\n")

            burst_alt = ahull.get_burst_altitude(safety_factor=p["SAFETY_FACTOR"])

            # 4. Get performance arrays
            h, Ln, Lg, I, BV, sigma, vol, surf_area = ahull.get_properties(n=100, include_tether=p["INCLUDE_TETHER"])

            operational_index = np.searchsorted(h, ahull.operational_altitude)

            P_op, T_op = get_atmospheric_properties(ahull.operational_altitude)
            T_env = get_thermal_model(T_op, p["SOLAR_FLUX"], p["ABSORPTIVITY"], p["EMISSIVITY"], p["WIND_SPEED"])

            max_rad = ahull.envelope.diameter / 2
            env_mass = surf_area * ahull.skin_density
            ballonet_fabric_mass = ahull.ballonet_fabric_mass * (vol ** (2/3))
            tether_mass_op = (ahull.tether_density * p["OPERATIONAL_HEIGHT"]) if p["INCLUDE_TETHER"] else 0
            gas_mass_op = get_gas_mass(P_op, T_op, vol, *ahull.gas_properties)

            # --- NEW: PIECEWISE WING STRUCTURAL MASS OVERRIDE ---
            if p.get("INCLUDE_WINGS", False) and p.get("WING_STATIONS"):
                stations = p["WING_STATIONS"]
                t_frac = p.get("WING_THICKNESS", 12.0) / 100.0
                density = p.get("FIN_DENSITY", 10.0)

                # A standard airfoil cross-sectional area is approx 0.685 * thickness * chord^2
                k_area = 0.685
                single_wing_vol = 0.0

                for i in range(len(stations) - 1):
                    dy = abs(stations[i+1]['y'] - stations[i]['y'])
                    c1 = stations[i]['chord']
                    c2 = stations[i+1]['chord']

                    # Exact geometric integration of the chord squared over the span segment (Frustum volume)
                    segment_vol = k_area * t_frac * (dy / 3.0) * (c1**2 + c1*c2 + c2**2)
                    single_wing_vol += segment_vol

                ahull.wing_mass = 2 * single_wing_vol * density # Account for both Left and Right wings
            # ----------------------------------------------------

            # --- UPDATED Total Mass Calculation ---
            total_mass_op = env_mass + ballonet_fabric_mass + tether_mass_op + ahull.fin_mass + ahull.wing_mass + ahull.additional_mass + gas_mass_op

            # --- UPDATED Total Mass Calculation ---
            total_mass_op = env_mass + ballonet_fabric_mass + tether_mass_op + ahull.fin_mass + ahull.wing_mass + ahull.additional_mass + gas_mass_op

            v_ballonet_total = BV[operational_index]
            v_per_ballonet = v_ballonet_total / max(p["BALLONET_NUMBER"], 1)
            ballonet_radius = (3 * v_per_ballonet / (4 * np.pi)) ** (1/3)

            op_stress = sigma[operational_index]
            allowable_stress = mat["base_strength"] / p["SAFETY_FACTOR"]
            safety_factor = mat["base_strength"] / np.max(sigma)

            print("  ")
            print("================ DESIGN PARAMETERS ===============")

            print(f"Profile Series:                     {series}")
            print(f"Hull Max Radius:                    {max_rad:.4f} m")
            print(f"Envelope Volume:                    {vol:.4f} m³")
            print(f"Envelope Surface Area:              {surf_area:.4f} m²\n")
            print("-" * 50)

            print(f"Ballonet Volume @ Op. Alt.:         {v_ballonet_total:.4f} m³")
            print(f"Ballonet Fabric Mass:               {ballonet_fabric_mass:.4f} kg\n")
            print(f"Effective Ballonet Radius:          {ballonet_radius:.4f} m")
            print("-" * 50)

            print(f"Envelope Fabric Mass:               {env_mass:.4f} kg")
            print(f"Fin mass:                           {ahull.fin_mass:.4f} kg")

            if p.get("INCLUDE_WINGS", False):
                print(f"Wing mass:                          {ahull.wing_mass:.4f} kg")

            print(f"Gas mass @ Op. Alt.:                {gas_mass_op:.4f} kg")
            print(f"Tether mass @ Op. Alt.:             {tether_mass_op:.4f} kg")
            print(f"Total mass @ Op. Alt.:              {total_mass_op:.4f} kg\n")
            print("-" * 50)

            print(f"Inflation Fraction @ Op. Alt.:      {I[operational_index]*100:.2f} %")
            print(f"Inflation Fraction @ Dep. Alt.:     {ahull.inflation_fraction_deploy*100:.2f} %\n")

            print("------------ STRESS & THERMAL ANALYSIS -----------")
            # print(f"Material Selected:                {p['MATERIAL_CLASS']}")
            print(f"Envelope Temp. @ Op. Alt.:          {T_env-273.15:.2f} °C")
            print(f"Estimated Burst Altitude:           {burst_alt:.2f} m")
            print(f"Total Combined Stress:              {op_stress:.2f} MPa")
            print(f"Allowable Stress:                   {allowable_stress:.2f} MPa")
            print(f"Safety Factor:                      {safety_factor:.2f}")
            print("="*50 + "\n")

            # --- INDEPENDENT MATERIAL LIFESPAN CALCULATION ---
            # Projecting strength decay over 10 years based on fatigue and UV
            t_years = np.linspace(0, 10, len(h))
            lifespan_strength = mat["base_strength"] * (p["FATIGUE_FACTOR"] ** t_years) * (1 - p["UV_DEGRADATION"] * t_years)

            # Store the expanded data and update 6-panel plots
            self.last_aero_data = (h, Ln, Lg, I, BV, sigma, t_years, lifespan_strength)
            self.update_aero_plots(*self.last_aero_data)

            self.log.append(f"[SUCCESS] Design parameters calculated for {ahull.envelope.length:.3f}m hull.")

        except Exception as e:
            self.log.append(f"[ERROR] Aerostat solver failed: {str(e)}")
            import traceback
            traceback.print_exc()

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

                geom = AirshipGeometry(params, self.salome_path, params['ENVELOPE_SERIES'])
                # Unpacking the five values: vol, surf, top, side, cv
                vol, surf, top, side, cv = geom.geometric_properties()

                self.prop_outputs["vol"].setText(f"{vol:.4f}")
                self.prop_outputs["surf"].setText(f"{surf:.4f}")
                self.prop_outputs["top_area"].setText(f"{top:.4f}")
                self.prop_outputs["side_area"].setText(f"{side:.4f}")

                # UPDATED: Format the CV coordinate pair
                self.prop_outputs["cv"].setText(f"{cv[0]:.2f}, {cv[1]:.2f}")
            except Exception as e:
                print(e)
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
            h, Ln, Lg, I, BV, sigma, t_years, lifespan_strength = self.last_aero_data
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Altitude (m)", "Net Lift (N)", "Gross Lift (N)",
                        "Inflation (%)", "Ballonet (m3)", "Stress (MPa)",
                        "Time (Years)", "Lifespan Strength (MPa)"
                    ])
                    for row in zip(h, Ln, Lg, I, BV, sigma, t_years, lifespan_strength):
                        writer.writerow(row)
                self.log.append(f"[SUCCESS] Exported to: {path}")
            except Exception as e:
                self.log.append(f"[ERROR] CSV Export failed: {e}")

    def get_parameters(self, target_dir):
        """
        Gathers all current GUI inputs into a single parameters dictionary.
        Optimization logic has been moved to run_instant_aerostat to prevent
        redundant class initialization.
        """
        p = {}
        all_keys = [
            "ENVELOPE_LENGTH", "ENVELOPE_RESOLUTION", "m1", "r0", "r1", "cp", "l2d",
            "FIN_AXIAL_OFFSET", "FIN_RC_LENGTH", "FIN_HEIGHT", "FIN_THICKNESS",
            "FIN_TAPER_RATIO", "FIN_SWEEP_ANGLE", "FIN_TIP_ANGLE", "FIN_NUMBER",
            "FIN_SECTION_RESOLUTION", "FIN_DENSITY", "THETA_RES", "PHI_RES", "ASPECT_RATIO",
            "BULGE_AMPLITUDE", "BULGE_POWER", "GORE_AMPLITUDE", "GORE_FADE_POWER",
            "OPERATIONAL_HEIGHT", "RELATIVE_HUMIDITY", "GAS_PURITY", "GAS_CONSTANT",
            "DELTA_P", "DELTA_T", "SKIN_DENSITY", "SKIN_THICKNESS", "PAYLOAD_MASS", "TARGET_NET_LIFT",
            "TETHER_DENSITY", "TETHER_FRACTION", "BALLONET_NUMBER",
            "BALLONET_FABRIC_DENSITY", "MARGIN_HEIGHT", "SAFETY_FACTOR",
            "SOLAR_FLUX", "WIND_SPEED", "EMISSIVITY", "ABSORPTIVITY",
            "FATIGUE_FACTOR", "UV_DEGRADATION", "WING_SPAN", "WING_ROOT_CHORD", "WING_TIP_CHORD", "WING_SWEEP",
            "WING_DIHEDRAL", "WING_TWIST_ROOT", "WING_TWIST_TIP",
            "WING_THICKNESS", "WING_AXIAL_OFFSET",
            "HULL_WIDTH", "HULL_HEIGHT" # Added Dragon Dream variables
        ]

        for key in all_keys:
            if key in self.inputs:
                p[key] = self.inputs[key].get_value()

        # Handle flatness percentage conversion safely
        if "BOTTOM_FLATNESS" in self.inputs:
            p["BOTTOM_FLATNESS"] = self.inputs["BOTTOM_FLATNESS"].get_value() / 100.0
        else:
            p["BOTTOM_FLATNESS"] = 0.25

        p["MATERIAL_CLASS"] = self.inputs["MATERIAL_CLASS"].currentText()
        p["INCLUDE_TETHER"] = self.inputs["INCLUDE_TETHER"].isChecked()
        p["BALLONET_SHAPE"] = self.inputs["BALLONET_SHAPE"].currentText()
        p["N_PETALS"] = self.inputs["N_PETALS"].get_value()
        p["LOBE_OFFSET_X"] = self.inputs["LOBE_OFFSET_X_SLIDER"].get_value()
        p["LOBE_OFFSET_Y"] = self.inputs["LOBE_OFFSET_Y_SLIDER"].get_value()
        p["LOBE_OFFSET_Z"] = self.inputs["LOBE_OFFSET_Z_SLIDER"].get_value()
        p["SHEET_LENGTH_RATIO"] = self.inputs["SHEET_LENGTH_RATIO_SLIDER"].get_value()
        p["LOBE_NUMBER"] = self.lobe_button_group.checkedId()
        p["FINAL_OBJECT_NAME"] = self.inputs["FINAL_OBJECT_NAME"].text()
        p["INCLUDE_FINS"] = self.inputs["INCLUDE_FINS"].isChecked()
        p["ENVELOPE_PARAMS"] = (p.get("m1", 0), p.get("r0", 0), p.get("r1", 0), p.get("cp", 0), p.get("l2d", 0))

        # --- CAPTURE WING TOGGLE ---
        if "INCLUDE_WINGS" in self.inputs:
            p["INCLUDE_WINGS"] = self.inputs["INCLUDE_WINGS"].isChecked()
        else:
            p["INCLUDE_WINGS"] = False

        # --- CAPTURE WING TOGGLE & STATIONS ---
        p["INCLUDE_WINGS"] = self.inputs.get("INCLUDE_WINGS") and self.inputs["INCLUDE_WINGS"].isChecked()

        if p["INCLUDE_WINGS"] and hasattr(self, 'wing_table'):
            try:
                stations = []
                airfoils_x = []
                airfoils_y = []

                # --- NEW: FORCE UNIFORM WING RESOLUTION ---
                # Read the resolution from the Root airfoil, fallback to 100
                global_wing_res = 100
                if hasattr(self, 'airfoil_library'):
                    root_combo = self.wing_table.cellWidget(0, 5)
                    root_name = root_combo.currentText() if root_combo else None
                    if root_name and root_name in self.airfoil_library:
                        global_wing_res = self.airfoil_library[root_name]["res"]

                for row in range(self.wing_table.rowCount()):
                    y_val = float(self.wing_table.item(row, 0).text())
                    chord = float(self.wing_table.item(row, 1).text())
                    twist = float(self.wing_table.item(row, 2).text())
                    x_off = float(self.wing_table.item(row, 3).text())
                    z_off = float(self.wing_table.item(row, 4).text())

                    combo = self.wing_table.cellWidget(row, 5)
                    af_name = combo.currentText() if combo else None

                    stations.append({'y': y_val, 'chord': chord, 'twist': twist, 'x_off': x_off, 'z_off': z_off})

                    # Read the exact pre-configured properties from the Asset Manager
                    if af_name and hasattr(self, 'airfoil_library') and af_name in self.airfoil_library:
                        af_data = self.airfoil_library[af_name]

                        # --- NEW: Override individual res with global_wing_res ---
                        res, scale = global_wing_res, af_data["scale"]
                        trans, rot = (af_data["tx"], af_data["ty"]), af_data["rot"]

                        if af_data["type"] == "FILE":
                            meth = None if "Auto" in af_data["method"] else af_data["method"]
                            x_pts, y_pts = get_airfoil_points(filename=af_data["path"], method=meth, resolution=res, scale_factor=scale, translation=trans, rotation_angle=rot)
                        else:
                            x_pts, y_pts = get_airfoil_points(thickness=af_data["thickness"], resolution=res, scale_factor=scale, translation=trans, rotation_angle=rot)
                    else:
                        # Absolute fallback
                        x_pts, y_pts = get_airfoil_points(thickness=12.0, resolution=100, scale_factor=1.0)

                    airfoils_x.append(x_pts.tolist() if hasattr(x_pts, 'tolist') else list(x_pts))
                    airfoils_y.append(y_pts.tolist() if hasattr(y_pts, 'tolist') else list(y_pts))
                # Package the compiled arrays directly into the parameter dictionary
                p["WING_STATIONS"] = stations
                p["WING_AIRFOILS_X"] = airfoils_x
                p["WING_AIRFOILS_Y"] = airfoils_y

            except Exception as e:
                self.log.append(f"[ERROR] Failed to compile wing stations: {e}")
                return None

        # Capture Profile Series
        p["ENVELOPE_SERIES"] = self.inputs["ENVELOPE_SERIES"].currentText()

        # Handle Volumetric scaling for geometry definition
        if self.mode_button_group.checkedId() == 2:
            from geometry_handler import GertlerEnvelope, NACAEnvelope, DragonDreamEnvelope

            if p["ENVELOPE_SERIES"] == "NACA":
                temp_env = NACAEnvelope.from_parameters((p["l2d"],), 1, int(p["ENVELOPE_RESOLUTION"]))
                temp_env.set_volume(
                    self.inputs["VOLUME"].get_value(),
                    p["LOBE_NUMBER"],
                    p["LOBE_OFFSET_X"], p["LOBE_OFFSET_Y"], p["LOBE_OFFSET_Z"]
                )
                p["ENVELOPE_LENGTH"] = temp_env.length
                self.inputs["ENVELOPE_LENGTH"].set_value(temp_env.length)

            elif p["ENVELOPE_SERIES"] == "GERTLER":
                temp_env = GertlerEnvelope.from_parameters_volume(
                    p["ENVELOPE_PARAMS"], self.inputs["VOLUME"].get_value(),
                    int(p["ENVELOPE_RESOLUTION"]), p["LOBE_NUMBER"],
                    p["LOBE_OFFSET_X"], p["LOBE_OFFSET_Y"], p["LOBE_OFFSET_Z"]
                )
                p["ENVELOPE_LENGTH"] = temp_env.length
                self.inputs["ENVELOPE_LENGTH"].set_value(temp_env.length)

            elif p["ENVELOPE_SERIES"] == "DRAGON_DREAM":
                # Tri-axial scaling logic. If you implement set_volume for DragonDream in the future, it goes here.
                pass

                # Determine Fin axial position based on length
        hull_len = p.get("ENVELOPE_LENGTH", 100)
        req_le = (p.get("FIN_AXIAL_OFFSET", 80) / 100.0) * hull_len
        max_le = hull_len - p.get("FIN_RC_LENGTH", 15) - 0.5
        p["FIN_AXIAL_OFFSET"] = min(req_le, max_le)

        try:
            theta_text = self.inputs["FIN_THETA_POS_TEXT"].text()
            p["FIN_THETA_POS"] = [float(a.strip()) for a in theta_text.split(',') if a.strip()]
        except:
            p["FIN_THETA_POS"] = [0, 90, 180, 270]

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

    def update_aero_plots(self, h, Ln, Lg, I, BV, sigma, t_years, lifespan_strength):
        """Updates the performance graphs using a 2x3 grid."""
        self.fig.clear()

        # Bundle data for dynamic plotting: (x_data, y_data, Title, X-Label, Color)
        datasets = [
            (h, Ln, "Net Static Lift (N)", "Altitude (m)", "#00BFFF"),
            (h, Lg, "Gross Static Lift (N)", "Altitude (m)", "#00FF00"),
            (h, I * 100, "Inflation Fraction (%)", "Altitude (m)", "#FF4500"),
            (h, BV, "Ballonet Volume (m³)", "Altitude (m)", "#DA70D6"),
            (h, sigma, "Total Envelope Stress (MPa)", "Altitude (m)", "#FFD700"),
            (t_years, lifespan_strength, "Material Lifespan (MPa)", "Time (Years)", "#FF6347")
        ]

        for i, (x, y, title, xlabel, color) in enumerate(datasets):
            ax = self.fig.add_subplot(2, 3, i + 1)
            ax.plot(x, y, color=color, linewidth=1.5)
            ax.set_title(title, color='#00BFFF', fontsize=10, fontweight='bold')
            ax.set_xlabel(xlabel, color='white', fontsize=8)
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
            self._update_3d_view(stl_path)

        if matrix is not None:
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
        """
        Resets all sliders across all tabs relative to current selections.
        Uses signal blocking to prevent the UI from freezing and ensures
        Fin Sweep and Configuration inputs are explicitly reset.
        """
        from geometry_handler import STANDARD_ENVELOPES

        current_tab = self.tab_widget.currentWidget()

        # BLOCK ALL UI SIGNALS TO PREVENT FREEZE
        for input_widget in self.inputs.values():
            if hasattr(input_widget, 'blockSignals'):
                input_widget.blockSignals(True)

        try:
            if current_tab == self.primary_input_tab:
                # Capture current state to preserve dropdowns
                current_shape_name = self.preset_combo.currentText()

                # Reset sliders to defaults of the CURRENTLY selected shape
                shape_vals = STANDARD_ENVELOPES.get(current_shape_name, STANDARD_ENVELOPES["NPL"])
                keys = ["m1", "r0", "r1", "cp", "l2d"]
                for k, v in zip(keys, shape_vals):
                    if k in self.inputs:
                        self.inputs[k].set_value(v)

                # Basic Geometry Defaults
                self.inputs["ENVELOPE_LENGTH"].set_value(100.0)
                self.inputs["VOLUME"].set_value(5000.0)
                self.inputs["ENVELOPE_RESOLUTION"].set_value(150)

                # Balloon & Wing Toggles
                if "INCLUDE_WINGS" in self.inputs:
                    self.inputs["INCLUDE_WINGS"].setChecked(False)

                # RESET SUPER PRESSURE BALLOON
                balloon_defaults = {
                    "ASPECT_RATIO": 1.0, "GORE_AMPLITUDE": 0.05, "GORE_FADE_POWER": 4.0,
                    "BULGE_AMPLITUDE": 0.0, "BULGE_POWER": 1.0, "THETA_RES": 400, "PHI_RES": 600
                }
                for key, val in balloon_defaults.items():
                    if key in self.inputs:
                        self.inputs[key].set_value(val)

            elif hasattr(self, 'aerostat_tab') and current_tab == self.aerostat_tab:
                aero_defaults = {
                    "OPERATIONAL_HEIGHT": 4500.0, "RELATIVE_HUMIDITY": 0.7, "MARGIN_HEIGHT": 500.0,
                    "GAS_PURITY": 0.97, "GAS_CONSTANT": 2077.0, "DELTA_P": 500.0, "DELTA_T": 5.0,
                    "BALLONET_NUMBER": 2, "BALLONET_FABRIC_DENSITY": 0.35, "SKIN_DENSITY": 0.75,
                    "SKIN_THICKNESS": 1.0, "PAYLOAD_MASS": 220.0, "TETHER_DENSITY": 0.1,
                    "TETHER_FRACTION": 1.0, "TARGET_NET_LIFT": 0.0, "SAFETY_FACTOR": 4.0,
                    "SOLAR_FLUX": 1000.0, "WIND_SPEED": 5.0, "EMISSIVITY": 0.8, "ABSORPTIVITY": 0.3,
                    "FATIGUE_FACTOR": 0.995, "UV_DEGRADATION": 0.02
                }
                for key, val in aero_defaults.items():
                    if key in self.inputs: self.inputs[key].set_value(val)
                self.inputs["MATERIAL_CLASS"].setCurrentIndex(0)
                self.inputs["INCLUDE_TETHER"].setChecked(True)
                self.inputs["OPTIMIZE_LENGTH"].setChecked(True)

            elif hasattr(self, 'fairings_tab') and current_tab == self.fairings_tab:
                lobe_offsets = {
                    "LOBE_OFFSET_X_SLIDER": 10.0, "LOBE_OFFSET_Y_SLIDER": 10.0, "LOBE_OFFSET_Z_SLIDER": 10.0,
                    "SHEET_LENGTH_RATIO_SLIDER": 0.5
                }
                for key, val in lobe_offsets.items():
                    if key in self.inputs: self.inputs[key].set_value(val)

            elif hasattr(self, 'fin_tab') and current_tab == self.fin_tab:
                fin_defaults = {
                    "FIN_RC_LENGTH": 15.5,
                    "FIN_HEIGHT": 15.5,
                    "FIN_THICKNESS": 10.0,
                    "FIN_TAPER_RATIO": 0.55,
                    "FIN_AXIAL_OFFSET": 80.0,
                    "FIN_SECTION_RESOLUTION": 60,
                    "FIN_SWEEP_ANGLE": 0.0,
                    "FIN_TIP_ANGLE": 15.0,
                    "FIN_NUMBER": 4,
                    "FIN_DENSITY": 10.0
                }
                for key, val in fin_defaults.items():
                    if key in self.inputs:
                        self.inputs[key].set_value(val)
                self.inputs["FIN_THETA_POS_TEXT"].setText("0.0, 90.0, 180.0, 270.0")
                self.inputs["INCLUDE_FINS"].setChecked(True)

            elif hasattr(self, 'wing_tab') and current_tab == self.wing_tab:
                if "WING_THICKNESS" in self.inputs: self.inputs["WING_THICKNESS"].set_value(12.0)
                if "WING_AXIAL_OFFSET" in self.inputs: self.inputs["WING_AXIAL_OFFSET"].set_value(40.0)

                # Clear the dynamic table and reset to standard root and tip
                self.wing_table.setRowCount(0)
                self._add_wing_station(0.0, 5.0, 2.0, 0.0, 0.0)
                self._add_wing_station(10.0, 2.0, -2.0, 2.58, 0.87)

            elif current_tab == self.output_tab:
                self.inputs["FINAL_OBJECT_NAME"].setText("Airship_Project")
                self.inputs["N_PETALS"].set_value(8)
                self.log.clear()
                for key in self.prop_outputs: self.prop_outputs[key].setText("0.0000")
                self.matrix_table.clearContents()
                self.plotter.clear()
                if hasattr(self, 'fig'):
                    self.fig.clear()
                    self.canvas.draw()
                self.inputs["COMPUTE_ADDED_MASS"].setChecked(True)

        finally:
            # Unblock signals and update UI once
            for input_widget in self.inputs.values():
                if hasattr(input_widget, 'blockSignals'):
                    input_widget.blockSignals(False)

            self.refresh_tabs()
            self._update_series_visibility() # Ensure correct slider state based on series
            self._auto_update_props()
            self.log.append(f"Status: Tab specific reset successful.")

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
