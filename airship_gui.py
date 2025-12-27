import sys
import os
import re
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSlider, QSizePolicy, QGroupBox,
    QMessageBox, QFileDialog, QTextEdit, QSpacerItem,
    QRadioButton, QButtonGroup, QCheckBox
)
from PySide6.QtGui import QFont, QDoubleValidator
from PySide6.QtCore import Qt, Signal

# --- HARDCODED PRESETS ---
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

try:
    from geometry import AirshipGeometry, calculate_petal_coordinates, plot_and_save_profile
except ImportError:
    AirshipGeometry = calculate_petal_coordinates = plot_and_save_profile = None

class LabeledSlider (QGroupBox):
    value_changed_by_user = Signal(float)
    def __init__(self, label, min_val, max_val, default_val, step, decimals=3, parent=None):
        super().__init__(label, parent)
        self.decimals, self.step = decimals, step
        self.min_val, self.max_val = min_val, max_val
        self._max_slider_val = int((max_val - min_val) / step)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self._max_slider_val)
        self.slider.setValue(int((default_val - min_val) / step))

        self.value_editor = QLineEdit()
        self.value_editor.setFixedWidth(70)
        self.value_editor.setFont(QFont("Monospace", 9))
        self.value_editor.setValidator(QDoubleValidator(min_val, max_val, decimals))

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.slider); h_layout.addWidget(self.value_editor)
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
        self.setGeometry(100, 100, 1200, 900)

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

        self.refresh_tabs()
        self.load_defaults()

    def _init_persistent_controls(self):
        self.mode_button_group = QButtonGroup(self)
        self.mode_btns = []
        for i, name in enumerate(["STANDARD MODE", "VOLUMETRIC MODE"], 1):
            btn = QPushButton(name); btn.setCheckable(True); btn.setMinimumHeight(40)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            self.mode_button_group.addButton(btn, i); self.mode_btns.append(btn)
        self.mode_btns[0].setChecked(True)

        self.lobe_button_group = QButtonGroup(self)
        self.lobe_btns = []
        for i, name in enumerate(["MONOLOBE", "BILOBE", "TRILOBE"], 1):
            btn = QPushButton(name); btn.setCheckable(True); btn.setMinimumHeight(40)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            self.lobe_button_group.addButton(btn, i); self.lobe_btns.append(btn)
        self.lobe_btns[0].setChecked(True)

        for btn in self.mode_btns: btn.clicked.connect(self.refresh_tabs)
        for btn in self.lobe_btns: btn.clicked.connect(self.refresh_tabs)

        self.header_widget = QWidget()
        h_layout = QVBoxLayout(self.header_widget); h_layout.setContentsMargins(0, 0, 0, 0)
        m_grp = QGroupBox("Dimensioning Mode"); m_lay = QHBoxLayout(m_grp)
        for btn in self.mode_btns: m_lay.addWidget(btn)
        l_grp = QGroupBox("Hull Configuration"); l_lay = QHBoxLayout(l_grp)
        for btn in self.lobe_btns: l_lay.addWidget(btn)
        h_layout.addWidget(m_grp); h_layout.addWidget(l_grp)

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #D4D4D4; font-family: Arial; font-size: 10pt; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background: #252526; }
            QTabBar::tab { background: #1e1e1e; padding: 10px 20px; border: 1px solid #3c3c3c; }
            QTabBar::tab:selected { background: #252526; border-top: 3px solid #00BFFF; }
            QGroupBox { border: 1px solid #3c3c3c; margin-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; color: #00BFFF; padding: 0 5px; }
            QLineEdit, QTextEdit { background-color: #3C3C3C; border: 1px solid #3c3c3c; color: #D4D4D4; }
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
        layout = QVBoxLayout(self.primary_input_tab); layout.addWidget(self.header_widget)
        self.hull_shape_box = QGroupBox("Hull Envelope Shape")
        cl = QGridLayout(self.hull_shape_box)
        self.preset_combo = QComboBox(); self.preset_combo.setObjectName("LargeDropdown")
        self.preset_combo.setMinimumHeight(45); self.preset_combo.addItems(list(STANDARD_ENVELOPES.keys()))
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        cl.addWidget(QLabel("Shape Preset:"), 0, 0); cl.addWidget(self.preset_combo, 0, 1, 1, 2)

        self.inputs["l2d"] = LabeledSlider("L/D Ratio", 1, 8, 3.266, 0.001)
        self.inputs["m1"] = LabeledSlider("m1", 0.3, 0.6, 0.419, 0.001)
        self.inputs["r0"] = LabeledSlider("r0", 0.01, 1, 0.337, 0.001)
        self.inputs["r1"] = LabeledSlider("r1", 0.01, 1, 0.251, 0.001)
        self.inputs["cp"] = LabeledSlider("cp", 0.5, 0.8, 0.651, 0.001)
        self.inputs["ENVELOPE_RESOLUTION"] = LabeledSlider("Resolution", 50, 500, 150, 1, 0)

        cl.addWidget(self.inputs["l2d"], 1, 0); cl.addWidget(self.inputs["m1"], 1, 1)
        cl.addWidget(self.inputs["r0"], 2, 0); cl.addWidget(self.inputs["r1"], 2, 1)
        cl.addWidget(self.inputs["cp"], 3, 0); cl.addWidget(self.inputs["ENVELOPE_RESOLUTION"], 3, 1)
        layout.addWidget(self.hull_shape_box)

        self.length_box = QGroupBox("Standard Mode: Length"); ll = QVBoxLayout(self.length_box); self.inputs["ENVELOPE_LENGTH"] = LabeledSlider("Length (L)", 10, 500, 100, 1, 1); ll.addWidget(self.inputs["ENVELOPE_LENGTH"]); layout.addWidget(self.length_box)
        self.volume_box = QGroupBox("Volumetric Mode: Volume"); vl = QVBoxLayout(self.volume_box); self.inputs["VOLUME"] = LabeledSlider("Volume (m³)", 100, 1000000, 5000, 10, 1); vl.addWidget(self.inputs["VOLUME"]); layout.addWidget(self.volume_box)
        layout.addStretch()

    def refresh_tabs(self):
        self.tab_widget.blockSignals(True)
        is_vol = self.mode_button_group.checkedId() == 2; is_multi = self.lobe_button_group.checkedId() > 1
        self.length_box.setHidden(is_vol); self.volume_box.setHidden(not is_vol)
        curr = self.tab_widget.currentIndex(); self.tab_widget.clear()
        self.tab_widget.addTab(self.primary_input_tab, "Envelope Geometry")
        if is_multi: self.tab_widget.addTab(self.fairings_tab, "Multi-Lobe Configuration")
        self.tab_widget.addTab(self.fin_tab, "Fin Design")
        self.tab_widget.addTab(self.output_tab, "Output")
        self.tab_widget.setCurrentIndex(min(curr, self.tab_widget.count()-1)); self.tab_widget.blockSignals(False)

    def setup_fairings_tab(self):
        layout = QVBoxLayout(self.fairings_tab)
        self.offset_box = QGroupBox("Lobe Separation Offsets")
        ol = QVBoxLayout(self.offset_box)
        self.inputs["LOBE_OFFSET_X_SLIDER"] = LabeledSlider("X Offset (Longitudinal)", 0, 50, 0, 0.1, 1)
        self.inputs["LOBE_OFFSET_Y_SLIDER"] = LabeledSlider("Y Offset (Lateral)", 0, 50, 0, 0.1, 1)
        self.inputs["LOBE_OFFSET_Z_SLIDER"] = LabeledSlider("Z Offset (Vertical)", 0, 50, 0, 0.1, 1)
        ol.addWidget(self.inputs["LOBE_OFFSET_X_SLIDER"])
        ol.addWidget(self.inputs["LOBE_OFFSET_Y_SLIDER"])
        ol.addWidget(self.inputs["LOBE_OFFSET_Z_SLIDER"])
        layout.addWidget(self.offset_box)

        sheet_grp = QGroupBox("Fairing Geometry")
        sl = QVBoxLayout(sheet_grp)
        self.inputs["SHEET_LENGTH_RATIO_SLIDER"] = LabeledSlider("Sheet Length Ratio (0-1)", 0, 1, 0.5, 0.01, 2)
        sl.addWidget(self.inputs["SHEET_LENGTH_RATIO_SLIDER"])
        layout.addWidget(sheet_grp)
        layout.addStretch()

    def setup_fin_tab(self):
        main_layout = QVBoxLayout(self.fin_tab)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 1. High Visibility Toggle First
        self.inputs["INCLUDE_FINS"] = QCheckBox("GENERATE FINS WITH HULL")
        self.inputs["INCLUDE_FINS"].setObjectName("FinToggle")
        self.inputs["INCLUDE_FINS"].setChecked(True)
        self.inputs["INCLUDE_FINS"].toggled.connect(self._toggle_fin_inputs)
        main_layout.addWidget(self.inputs["INCLUDE_FINS"])

        # 2. Grouped Dimensions
        self.fin_container = QWidget()
        container_layout = QVBoxLayout(self.fin_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        fin_dim_group = QGroupBox("Fin Dimensions")
        fin_dim_layout = QGridLayout(fin_dim_group)
        self.inputs["FIN_RC_LENGTH"] = LabeledSlider("Root Chord Length", 5.0, 50.0, 15.5, 0.1, 1)
        self.inputs["FIN_HEIGHT"] = LabeledSlider("Fin Height (Span)", 5.0, 50.0, 15.5, 0.1, 1)
        self.inputs["FIN_THICKNESS"] = LabeledSlider("Thickness (% of RC)", 1.0, 20.0, 10.0, 0.1, 1)
        self.inputs["FIN_TAPER_RATIO"] = LabeledSlider("Taper Ratio (Tip/Root)", 0.1, 1.0, 0.55, 0.01, 2)
        self.inputs["FIN_AXIAL_OFFSET"] = LabeledSlider("Axial Offset (% Length)", 50.0, 100.0, 80.0, 0.1, 1)
        self.inputs["FIN_SECTION_RESOLUTION"] = LabeledSlider("Section Resolution", 10, 100, 60, 1, decimals=0)

        fin_dim_layout.addWidget(self.inputs["FIN_RC_LENGTH"], 0, 0); fin_dim_layout.addWidget(self.inputs["FIN_HEIGHT"], 0, 1); fin_dim_layout.addWidget(self.inputs["FIN_THICKNESS"], 0, 2)
        fin_dim_layout.addWidget(self.inputs["FIN_TAPER_RATIO"], 1, 0); fin_dim_layout.addWidget(self.inputs["FIN_AXIAL_OFFSET"], 1, 1); fin_dim_layout.addWidget(self.inputs["FIN_SECTION_RESOLUTION"], 1, 2)
        container_layout.addWidget(fin_dim_group)

        fin_sweep_group = QGroupBox("Fin Sweep and Configuration")
        fin_sweep_layout = QGridLayout(fin_sweep_group)
        self.inputs["FIN_SWEEP_ANGLE"] = LabeledSlider("Sweep Angle (Deg)", 0.0, 45.0, 0.0, 0.1, 1)
        self.inputs["FIN_TIP_ANGLE"] = LabeledSlider("Tip Angle (Deg)", 0.0, 30.0, 15.0, 0.1, 1)
        self.inputs["FIN_NUMBER"] = LabeledSlider("N Fins", 2, 8, 4, 1, decimals=0)

        fin_sweep_layout.addWidget(self.inputs["FIN_SWEEP_ANGLE"], 0, 0); fin_sweep_layout.addWidget(self.inputs["FIN_TIP_ANGLE"], 0, 1)
        fin_sweep_layout.addWidget(QLabel("Number of Fins:"), 1, 0); fin_sweep_layout.addWidget(self.inputs["FIN_NUMBER"], 1, 1)
        fin_sweep_layout.addWidget(QLabel("Angular Positions (Comma Separated):"), 2, 0)
        self.inputs["FIN_THETA_POS_TEXT"] = QLineEdit("0.0, 90.0, 180.0, 270.0")
        fin_sweep_layout.addWidget(self.inputs["FIN_THETA_POS_TEXT"], 2, 1, 1, 2)
        container_layout.addWidget(fin_sweep_group)

        main_layout.addWidget(self.fin_container)
        main_layout.addStretch(1)

    def _toggle_fin_inputs(self, enabled):
        """Disables all fin input sliders and text editors."""
        self.fin_container.setEnabled(enabled)

    def setup_output_tab(self):
        layout = QVBoxLayout(self.output_tab)
        self.inputs["FINAL_OBJECT_NAME"] = QLineEdit("Airship_Project")
        layout.addWidget(QLabel("Project Name:")); layout.addWidget(self.inputs["FINAL_OBJECT_NAME"])

        dir_group = QGroupBox("Base Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        self.dir_path_display = QLineEdit(self.base_output_directory)
        self.dir_path_display.setReadOnly(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_output_directory)
        dir_layout.addWidget(self.dir_path_display); dir_layout.addWidget(btn_browse)
        layout.addWidget(dir_group)

        self.inputs["N_PETALS"] = LabeledSlider("Gores/Petals", 2, 200, 8, 1, 0)
        layout.addWidget(self.inputs["N_PETALS"])

        self.format_button_group = QButtonGroup(self); h_lay = QHBoxLayout()
        for i, fmt in enumerate([".brep", ".stl", ".step"]):
            btn = QPushButton(fmt); btn.setCheckable(True); h_lay.addWidget(btn); self.format_button_group.addButton(btn, i); btn.setChecked(i==1)
        layout.addLayout(h_lay)

        btn_lay = QHBoxLayout()
        self.btn_run = QPushButton("RUN GENERATION")
        self.btn_run.setMinimumHeight(35); self.btn_run.setStyleSheet("background-color: #007ACC; color: white;")
        self.btn_run.clicked.connect(self.run_process)

        self.btn_plot = QPushButton("PLOT 2D PETAL")
        self.btn_plot.setMinimumHeight(35); self.btn_plot.setEnabled(False)
        self.btn_plot.clicked.connect(self.generate_plot)

        btn_lay.addWidget(self.btn_run); btn_lay.addWidget(self.btn_plot); layout.addLayout(btn_lay)
        self.log = QTextEdit("Status: Ready"); self.log.setReadOnly(True); layout.addWidget(self.log)

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
        for key in slider_keys:
            if key in self.inputs: p[key] = self.inputs[key].get_value()
        p["N_PETALS"] = self.inputs["N_PETALS"].get_value()
        p["LOBE_OFFSET_X"] = self.inputs["LOBE_OFFSET_X_SLIDER"].get_value()
        p["LOBE_OFFSET_Y"] = self.inputs["LOBE_OFFSET_Y_SLIDER"].get_value()
        p["LOBE_OFFSET_Z"] = self.inputs["LOBE_OFFSET_Z_SLIDER"].get_value()
        p["MULTI_LOBE_OFFSET_FACTOR"] = 0
        p["SHEET_LENGTH_RATIO"] = self.inputs["SHEET_LENGTH_RATIO_SLIDER"].get_value()
        lobe_id = self.lobe_button_group.checkedId()
        p["LOBE_NUMBER"] = lobe_id if lobe_id != -1 else 1
        p["ENVELOPE_PARAMS"] = (p["m1"], p["r0"], p["r1"], p["cp"], p["l2d"])
        p["FINAL_OBJECT_NAME"] = self.inputs["FINAL_OBJECT_NAME"].text()
        p["type"] = self.preset_combo.currentText().split(" ")[0]
        p["INCLUDE_FINS"] = self.inputs["INCLUDE_FINS"].isChecked()

        is_vol = self.mode_button_group.checkedId() == 2
        hull_len = p.get("ENVELOPE_LENGTH", 100.0)
        if is_vol:
            try:
                from geometry_handler import GertlerEnvelope
                temp_env = GertlerEnvelope.from_parameters_volume(p["ENVELOPE_PARAMS"], self.inputs["VOLUME"].get_value(), int(p["ENVELOPE_RESOLUTION"]), p["LOBE_NUMBER"], p["LOBE_OFFSET_X"], p["LOBE_OFFSET_Y"], p["LOBE_OFFSET_Z"])
                hull_len = temp_env.length
                p["ENVELOPE_LENGTH"] = hull_len
            except: hull_len = 100.0
        req_le = (p["FIN_AXIAL_OFFSET"] / 100.0) * hull_len
        max_le = hull_len - p["FIN_RC_LENGTH"] - 0.5
        p["FIN_AXIAL_OFFSET"] = min(req_le, max_le)
        theta_text = self.inputs["FIN_THETA_POS_TEXT"].text()
        try:
            p["FIN_THETA_POS"] = [float(a.strip()) for a in theta_text.split(',') if a.strip()]
        except ValueError: return None
        p["OUTPUT_DIRECTORY"] = target_dir
        return p

    def run_process(self):
        if AirshipGeometry is None: return
        target_dir = self.create_new_output_folder()
        p = self.get_parameters(target_dir)
        if p is None: return
        fmt_idx = self.format_button_group.checkedId()
        fmt_ext = [".brep", ".stl", ".step"][fmt_idx]; fmt_name = ["BREP", "STL", "STEP"][fmt_idx]
        export_path = os.path.join(target_dir, p["FINAL_OBJECT_NAME"] + fmt_ext)
        try:
            g = AirshipGeometry(p, self.salome_path)
            original_generate = g._generate_salome_script
            def patched_generate(export_file, export_format, open_gui):
                path = original_generate(export_path, export_format, open_gui)
                with open(path, 'r') as f: content = f.read()
                if export_format == "STL":
                    stl_export_logic = (f"\n# --- User STL Export ---\nimport GEOM\n"
                                        f"geompy.ExportSTL(Final_Hull_Solid, r'{export_path}', False)\n"
                                        f"print('Exported STL successfully.')\n")
                    content += stl_export_logic
                with open(path, 'w') as f: f.write(content)
                return path
            g._generate_salome_script = patched_generate
            g.run_salome(False, p["FINAL_OBJECT_NAME"] + fmt_ext, fmt_name)
            self.btn_plot.setEnabled(True)
            self.log.append(f"Geometry saved in: {target_dir}")
        except Exception as e: self.log.append(f"Error: {e}")

    def generate_plot(self):
        if calculate_petal_coordinates is None: return
        target_dir = self.current_session_folder
        p = self.get_parameters(target_dir)
        if p is None: return
        try:
            L = p.get("ENVELOPE_LENGTH", 100.0)
            coords, nx, nc = calculate_petal_coordinates(p, L, int(p["N_PETALS"]))
            dat_file = os.path.join(target_dir, p["FINAL_OBJECT_NAME"]+".dat")
            msg = plot_and_save_profile(coords, p["FINAL_OBJECT_NAME"], dat_file, nx, nc, int(p["N_PETALS"]))
            self.log.append(f"Plot saved in: {target_dir}")
        except Exception as e: self.log.append(f"Plot Error: {e}")

    def load_defaults(self): self.load_preset(0)
    def load_preset(self, idx):
        vals = STANDARD_ENVELOPES[self.preset_combo.itemText(idx)]
        for k, v in zip(["m1", "r0", "r1", "cp", "l2d"], vals):
            if k in self.inputs: self.inputs[k].set_value(v)

    def reset_to_defaults(self):
        self.load_preset(self.preset_combo.currentIndex())
        self.log.append("Parameters reset.")

    def setup_navigation_buttons(self):
        nav = QWidget(); lay = QHBoxLayout(nav)
        self.btn_back = QPushButton("<< PREV"); self.btn_back.setMinimumHeight(35); self.btn_back.setFixedWidth(100)
        self.btn_next = QPushButton("NEXT >>"); self.btn_next.setMinimumHeight(35); self.btn_next.setFixedWidth(100)
        self.btn_next.setStyleSheet("background-color: #00BFFF; color: black; font-weight: bold;")
        self.btn_back.clicked.connect(lambda: self.tab_widget.setCurrentIndex(self.tab_widget.currentIndex()-1))
        self.btn_next.clicked.connect(lambda: self.tab_widget.setCurrentIndex(self.tab_widget.currentIndex()+1))
        self.btn_reset = QPushButton("RESET"); self.btn_reset.setMinimumHeight(35); self.btn_reset.setFixedWidth(80)
        self.btn_reset.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        self.btn_exit = QPushButton("EXIT"); self.btn_exit.setMinimumHeight(35); self.btn_exit.setFixedWidth(80)
        self.btn_exit.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold;")
        self.btn_exit.clicked.connect(self.close)
        lay.addWidget(self.btn_back); lay.addWidget(self.btn_next); lay.addStretch()
        lay.addWidget(self.btn_reset); lay.addWidget(self.btn_exit)
        return nav

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AirshipGUI()
    ex.show()
    sys.exit(app.exec())
