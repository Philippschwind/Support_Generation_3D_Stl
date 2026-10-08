import sys
import json
from pathlib import Path

import pyvista as pv

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QMessageBox,
)

from pyvistaqt import QtInteractor


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Support Generator")
        self.resize(1200, 750)

        self.mesh = None
        self.file_path = None

        # Stores dynamically created parameter widgets
        self.parameter_widgets = {}

        self.setup_ui()
        self.load_parameter_config("parameters.json")

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # =========================
        # LEFT: 3D Viewer
        # =========================

        viewer_layout = QVBoxLayout()

        self.plotter = QtInteractor(self)

        viewer_layout.addWidget(self.plotter.interactor)

        self.file_label = QLabel("No STL file selected")
        viewer_layout.addWidget(self.file_label)

        # =========================
        # RIGHT: Controls
        # =========================

        control_widget = QWidget()
        control_widget.setMaximumWidth(350)

        control_layout = QVBoxLayout(control_widget)

        # STL selection

        self.open_button = QPushButton("Open STL")
        self.open_button.clicked.connect(self.open_stl)

        control_layout.addWidget(self.open_button)

        # Parameter area

        parameter_group = QGroupBox("Support Parameters")

        self.parameter_layout = QFormLayout(parameter_group)

        control_layout.addWidget(parameter_group)

        # Generate button

        self.generate_button = QPushButton("Generate Support")
        self.generate_button.clicked.connect(self.generate_support)

        control_layout.addWidget(self.generate_button)

        control_layout.addStretch()

        # =========================

        main_layout.addLayout(viewer_layout, 1)
        main_layout.addWidget(control_widget)

    # ---------------------------------------------------------
    # JSON Parameter Loading
    # ---------------------------------------------------------

    def load_parameter_config(self, filename):

        path = Path(filename)

        if not path.exists():
            QMessageBox.warning(
                self,
                "Error",
                f"Parameter file not found:\n{filename}"
            )
            return

        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)

        for parameter_name, parameter_config in config.items():

            widget = self.create_parameter_widget(parameter_config)

            if widget is None:
                continue

            self.parameter_widgets[parameter_name] = widget

            label = parameter_config.get(
                "label",
                parameter_name
            )

            self.parameter_layout.addRow(label, widget)

    # ---------------------------------------------------------
    # Create widgets dynamically
    # ---------------------------------------------------------

    def create_parameter_widget(self, config):

        parameter_type = config.get("type")

        # FLOAT
        if parameter_type == "float":

            widget = QDoubleSpinBox()

            widget.setRange(
                config.get("min", -1000000),
                config.get("max", 1000000)
            )

            widget.setValue(
                config.get("default", 0.0)
            )

            widget.setSingleStep(
                config.get("step", 0.1)
            )

            widget.setDecimals(3)

            return widget

        # INTEGER
        elif parameter_type == "int":

            widget = QSpinBox()

            widget.setRange(
                config.get("min", -1000000),
                config.get("max", 1000000)
            )

            widget.setValue(
                config.get("default", 0)
            )

            return widget

        # BOOLEAN
        elif parameter_type == "bool":

            widget = QCheckBox()

            widget.setChecked(
                config.get("default", False)
            )

            return widget

        else:

            print(
                f"Unknown parameter type: {parameter_type}"
            )

            return None

    # ---------------------------------------------------------
    # STL Loading
    # ---------------------------------------------------------

    def open_stl(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open STL",
            "",
            "STL Files (*.stl)"
        )

        if not filename:
            return

        try:

            mesh = pv.read(filename)

            self.mesh = mesh
            self.file_path = filename

            self.file_label.setText(
                Path(filename).name
            )

            self.show_mesh()

        except Exception as error:

            QMessageBox.critical(
                self,
                "Error",
                f"Could not load STL:\n{error}"
            )

    # ---------------------------------------------------------
    # Display Mesh
    # ---------------------------------------------------------

    def show_mesh(self):

        if self.mesh is None:
            return

        self.plotter.clear()

        self.plotter.add_mesh(
            self.mesh,
            show_edges=True
        )

        self.plotter.reset_camera()

    # ---------------------------------------------------------
    # Read parameter values
    # ---------------------------------------------------------

    def get_parameters(self):

        parameters = {}

        for name, widget in self.parameter_widgets.items():

            if isinstance(widget, QDoubleSpinBox):
                parameters[name] = widget.value()

            elif isinstance(widget, QSpinBox):
                parameters[name] = widget.value()

            elif isinstance(widget, QCheckBox):
                parameters[name] = widget.isChecked()

        return parameters

    # ---------------------------------------------------------
    # Support Generation
    # ---------------------------------------------------------

    def generate_support(self):

        if self.mesh is None:

            QMessageBox.warning(
                self,
                "No STL",
                "Please select an STL file first."
            )

            return

        parameters = self.get_parameters()

        print("Input STL:")
        print(self.file_path)

        print("\nParameters:")

        for key, value in parameters.items():
            print(f"{key}: {value}")

        # =====================================================
        # YOUR SUPPORT GENERATION CODE GOES HERE
        # =====================================================

        # Example:
        #
        # support_mesh = generate_support(
        #     self.mesh,
        #     parameters
        # )
        #
        # self.plotter.add_mesh(
        #     support_mesh,
        #     color="red"
        # )

        # =====================================================

        QMessageBox.information(
            self,
            "Support Generator",
            "Parameters loaded successfully.\n"
            "Support generation would start now."
        )


# -------------------------------------------------------------
# Application Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())