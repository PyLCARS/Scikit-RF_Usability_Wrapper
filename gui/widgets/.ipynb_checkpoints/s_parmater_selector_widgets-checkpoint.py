# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S-Parameter Selector Widgets
#
# Interactive button-based selectors for S-parameters in RF networks.
#
# ## Widgets
# - **PortSelector**: General N×N port selector for any touchstone file
# - **DifferentialPortSelector**: 4-port differential/common mode selector
#
# ## Features
# - Toggle selection with visual feedback
# - Programmatic selection control via `set_selected_ports()`
# - Signal emission for integration with plotting/analysis
#
# Initial Author: GProtoZeroW - Jan 2026

# %% [markdown]
# ## Setup Qt Event Loop

# %%
# %gui qt

# %% [markdown]
# ## Imports

# %%
from pathlib import Path
import skrf as rf

# %% [markdown]
# ## Path Setup and Dependencies

# %%
# Setup path - go up to project root
import sys
from pathlib import Path

# Current dir is gui/widgets, go up 2 levels to project root
project_root = Path.cwd().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# PySide6 imports
from PySide6 import QtWidgets, QtCore, QtGui

# Import base class and enums
from gui.widget_utilities import WidgetUtilities
from core.scikit_rf_enums import ParameterType

# Get or create QApplication
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

print(f"✓ Project root: {project_root}")
print(f"✓ Imports successful")

# %% [markdown]
# ## Load Example Touchstone File
#
# Load a 6-port example for testing the general port selector.

# %%
# Path to example touchstone file
example_file_s6p_path = project_root / "exsample_touchstones" / "from_scikit-rf" / "cst_example_6ports_V2.s6p"

# Verify it exists
if example_file_s6p_path.exists():
    print(f"✓ Found: {example_file_s6p_path.name}")
    print(f"  Full path: {example_file_s6p_path}")
else:
    print(f"✗ File not found: {example_file_s6p_path}")

# %%
s6p_network = rf.Network(example_file_s6p_path)

# %%
# Show available port combinations
print(f"Network has {s6p_network.nports} ports")
print(f"Total port combinations: {len(s6p_network.port_tuples)}")
print(f"First few: {s6p_network.port_tuples[:6]}")

# %% [markdown]
# ## PortSelector Widget
#
# General N×N grid selector for S-parameters. Works with any touchstone file.
#
# ### Features
# - Auto-sizes grid based on number of ports
# - Buttons labeled S11, S21, etc. (1-indexed display)
# - Emits 0-indexed tuples: `(row, col)`
# - Toggle selection with click
# - Green highlight for selected parameters

# %%
class PortSelector(QtWidgets.QWidget, WidgetUtilities):
    """Calculator-style button grid for selecting S-parameter port pairs.
    
    Creates an N×N grid of toggle buttons for selecting S-parameters from
    any RF network. Buttons display 1-indexed labels (S11, S21, etc.) but
    emit 0-indexed tuples for programmatic use.
    
    Signals:
        port_selected(tuple): Emitted when a port is selected, tuple is (row, col)
        port_deselected(tuple): Emitted when a port is deselected
        selection_changed(set): Emitted with complete set of selected ports
    
    Examples:
        >>> selector = PortSelector(network.port_tuples)
        >>> selector.port_selected.connect(lambda p: print(f"Selected: {p}"))
        >>> selector.show()
        >>> 
        >>> # Programmatic selection
        >>> selector.set_selected_ports([(0, 0), (1, 1), (2, 2)])
    """
    
    port_selected = QtCore.Signal(tuple)      # (row, col) 0-indexed
    port_deselected = QtCore.Signal(tuple)    # (row, col) 0-indexed
    selection_changed = QtCore.Signal(set)    # Set of all selected tuples
    
    def __init__(self, port_tuples, parent=None):
        """Initialize port selector.
        
        Args:
            port_tuples: List of (row, col) tuples from rf.Network.port_tuples
            parent: Optional parent widget
        """
        self.port_tuples = port_tuples
        
        # Determine grid dimensions
        max_row = max(t[0] for t in port_tuples)
        max_col = max(t[1] for t in port_tuples)
        self.n_rows = max_row + 1
        self.n_cols = max_col + 1
        
        # Storage
        self.buttons = {}
        self.selected_ports = set()
        
        super().__init__(parent)
        self._widget_startup_init_calls()
    
    def gui_layout_init(self):
        """Create N×N button grid."""
        layout = QtWidgets.QGridLayout(self)
        
        for row_idx in range(self.n_rows):
            for col_idx in range(self.n_cols):
                # Display as 1-indexed: S11, S21, S32, etc.
                button_text = f"S{row_idx+1}{col_idx+1}"
                button = QtWidgets.QPushButton(button_text)
                button.setCheckable(True)
                
                # Store 0-indexed tuple for programmatic use
                button.setProperty("port_tuple", (row_idx, col_idx))
                
                layout.addWidget(button, row_idx, col_idx)
                self.buttons[(row_idx, col_idx)] = button
    
    def gui_sizing_init(self):
        """Set button sizes."""
        for button in self.buttons.values():
            button.setFixedSize(60, 40)
    
    def gui_styling_init(self):
        """Apply button styling with selection state."""
        self.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #999;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #45a049;
            }
            QPushButton:checked:hover {
                background-color: #45a049;
            }
        """)
    
    def gui_wiring_init(self):
        """Connect button signals."""
        for button in self.buttons.values():
            button.toggled.connect(self.on_button_toggled)
    
    def on_button_toggled(self, checked):
        """Handle button toggle event.
        
        Args:
            checked: True if button was checked, False if unchecked
        """
        button = self.sender()
        port_tuple = button.property("port_tuple")
        
        if checked:
            self.selected_ports.add(port_tuple)
            print(f"Selected: S{port_tuple[0]+1}{port_tuple[1]+1} → {port_tuple}")
            self.port_selected.emit(port_tuple)
        else:
            self.selected_ports.discard(port_tuple)
            print(f"Deselected: S{port_tuple[0]+1}{port_tuple[1]+1}")
            self.port_deselected.emit(port_tuple)
        
        self.selection_changed.emit(self.selected_ports.copy())
    
    def set_selected_ports(self, port_tuples):
        """Set selected ports programmatically (e.g., from plot state).
        
        Args:
            port_tuples: Iterable of (row, col) tuples to select
            
        Examples:
            >>> selector.set_selected_ports([(0, 0), (1, 1)])
            >>> selector.set_selected_ports([])  # Clear all
        """
        with self.temp_block_widgets_signals(*self.buttons.values()):
            # Clear all buttons
            for button in self.buttons.values():
                button.setChecked(False)
            
            # Set new selection
            self.selected_ports = set(port_tuples)
            for port_tuple in port_tuples:
                if port_tuple in self.buttons:
                    self.buttons[port_tuple].setChecked(True)
        
        # Emit change signal (not blocked)
        self.selection_changed.emit(self.selected_ports.copy())
    
    def get_selected_ports(self):
        """Get currently selected port tuples.
        
        Returns:
            set: Copy of selected port tuples
        """
        return self.selected_ports.copy()
    
    def clear_selection(self):
        """Clear all selected ports."""
        self.set_selected_ports([])


# %% [markdown]
# ## Test PortSelector Widget

# %%
# Create and show selector
selector = PortSelector(s6p_network.port_tuples)
selector.setWindowTitle("S-Parameter Port Selector")
selector.show()

# Example: Programmatically select diagonal
# selector.set_selected_ports([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

# %% [markdown]
# ## DifferentialPortSelector Widget
#
# Specialized 4-port selector for differential/common mode analysis.
#
# ### Layout
# - **2×2 block arrangement** by mode type (DD, DC, CD, CC)
# - Each block contains **2×2 buttons** for port pairs (11, 12, 21, 22)
# - Uses **ijmode indexing**: `(mode, i, j)` where mode ∈ {DD, DC, CD, CC}
#
# ### Visual Cues
# - **Red**: DD and CC (pure differential/common modes)
# - **Black**: DC and CD (mode conversion - typically undesirable)
# - Descriptive labels explain signal flow

# %%
class DifferentialPortSelector(QtWidgets.QWidget, WidgetUtilities):
    """4-port differential parameter selector using mixed-mode layout.
    
    Organized as four quadrants (DD, DC, CD, CC) each containing a 2×2
    grid of port pair combinations. Designed for analyzing differential
    pairs in high-speed digital designs.
    
    Signals:
        port_selected(tuple): Emitted when selected, tuple is (mode, i, j)
        port_deselected(tuple): Emitted when deselected
        selection_changed(set): Complete set of selected (mode, i, j) tuples
    
    Examples:
        >>> selector = DifferentialPortSelector()
        >>> selector.show()
        >>> 
        >>> # Select differential parameters
        >>> selector.set_selected_ports([('DD', 1, 1), ('DD', 2, 2)])
    """
    
    port_selected = QtCore.Signal(tuple)      # (mode, i, j)
    port_deselected = QtCore.Signal(tuple)    # (mode, i, j)
    selection_changed = QtCore.Signal(set)    # Set of (mode, i, j) tuples
    
    def __init__(self, parent=None):
        """Initialize differential port selector.
        
        Args:
            parent: Optional parent widget
        """
        self.buttons = {}
        self.selected_ports = set()
        
        QtWidgets.QWidget.__init__(self, parent)
        self._widget_startup_init_calls()
    
    def gui_layout_init(self):
        """Create 2×2 block layout with 2×2 button grids in each block."""
        main_layout = QtWidgets.QGridLayout(self)
        main_layout.setSpacing(15)
        
        # Define quadrants: (grid_row, grid_col, mode, description, label_color)
        quadrants = [
            (0, 0, 'DD', 'Differential in,\ndifferential out', '#c62828'),
            (0, 1, 'DC', 'Differential in,\ncommon out', '#333333'),
            (1, 0, 'CD', 'Common in,\ndifferential out', '#333333'),
            (1, 1, 'CC', 'Common in,\ncommon out', '#c62828'),
        ]
        
        # Unicode subscript mapping for button text
        subscript_digits = {'1': '₁', '2': '₂', '3': '₃', '4': '₄', 
                           '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
        subscript_letters = {'D': 'ᴅ', 'C': 'ᴄ'}
        
        for quad_row, quad_col, mode, label_text, color in quadrants:
            # Frame for this mode block
            frame = QtWidgets.QWidget()
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setSpacing(2)
            
            # Mode label at top
            mode_label = QtWidgets.QLabel(mode)
            mode_label.setStyleSheet(f"""
                font-weight: bold; 
                font-size: 14px; 
                color: {color};
                padding: 3px;
            """)
            mode_label.setAlignment(QtCore.Qt.AlignCenter)
            frame_layout.addWidget(mode_label)
            
            # 2×2 grid of buttons for port pairs
            button_grid = QtWidgets.QGridLayout()
            button_grid.setSpacing(2)
            
            for i in range(1, 3):  # Pair index 1, 2
                for j in range(1, 3):  # Pair index 1, 2
                    # Create subscripted text: S₁₁ᴅᴅ, S₁₂ᴅᴄ, etc.
                    i_sub = subscript_digits[str(i)]
                    j_sub = subscript_digits[str(j)]
                    mode_sub = ''.join(subscript_letters[c] for c in mode)
                    button_text = f"S{i_sub}{j_sub}{mode_sub}"
                    
                    button = QtWidgets.QPushButton(button_text)
                    button.setCheckable(True)
                    button.setProperty("port_tuple", (mode, i, j))
                    
                    button_grid.addWidget(button, i-1, j-1)
                    self.buttons[(mode, i, j)] = button
            
            frame_layout.addLayout(button_grid)
            
            # Description label at bottom
            desc_label = QtWidgets.QLabel(label_text)
            desc_label.setStyleSheet("font-size: 9px; color: #666;")
            desc_label.setAlignment(QtCore.Qt.AlignCenter)
            desc_label.setWordWrap(True)
            frame_layout.addWidget(desc_label)
            
            # Style the frame
            frame.setStyleSheet("""
                QWidget {
                    background-color: #f5f5f5;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 5px;
                }
            """)
            
            main_layout.addWidget(frame, quad_row, quad_col)
    
    def gui_sizing_init(self):
        """Set button sizes."""
        for button in self.buttons.values():
            button.setFixedSize(65, 35)
    
    def gui_styling_init(self):
        """Apply mode-specific styling to buttons."""
        base_style = """
            QPushButton {
                background-color: white;
                border: 1px solid #999;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:checked {
                color: white;
                border: 2px solid #333;
            }
        """
        
        for (mode, i, j), button in self.buttons.items():
            if mode in ('DD', 'CC'):
                # Pure modes - red when selected
                button.setStyleSheet(base_style + """
                    QPushButton:checked {
                        background-color: #c62828;
                        border-color: #b71c1c;
                    }
                """)
            else:
                # Mode conversion (DC, CD) - black when selected
                button.setStyleSheet(base_style + """
                    QPushButton:checked {
                        background-color: #333333;
                        border-color: #000000;
                    }
                """)
    
    def gui_wiring_init(self):
        """Connect button signals."""
        for button in self.buttons.values():
            button.toggled.connect(self.on_button_toggled)
    
    def on_button_toggled(self, checked):
        """Handle button toggle event.
        
        Args:
            checked: True if button was checked, False if unchecked
        """
        button = self.sender()
        ijmode_tuple = button.property("port_tuple")
        mode, i, j = ijmode_tuple
        
        if checked:
            self.selected_ports.add(ijmode_tuple)
            conversion_note = " ⚠ mode conversion" if mode in ('DC', 'CD') else ""
            print(f"Selected: S{i}{j}{mode}{conversion_note} → {ijmode_tuple}")
            self.port_selected.emit(ijmode_tuple)
        else:
            self.selected_ports.discard(ijmode_tuple)
            print(f"Deselected: S{i}{j}{mode}")
            self.port_deselected.emit(ijmode_tuple)
        
        self.selection_changed.emit(self.selected_ports.copy())
    
    def set_selected_ports(self, port_tuples):
        """Set selected ports programmatically.
        
        Args:
            port_tuples: Iterable of (mode, i, j) tuples to select
            
        Examples:
            >>> selector.set_selected_ports([('DD', 1, 1), ('DD', 2, 2)])
            >>> selector.set_selected_ports([])  # Clear all
        """
        with self.temp_block_widgets_signals(*self.buttons.values()):
            # Clear all buttons
            for button in self.buttons.values():
                button.setChecked(False)
            
            # Set new selection
            self.selected_ports = set(port_tuples)
            for port_tuple in port_tuples:
                if port_tuple in self.buttons:
                    self.buttons[port_tuple].setChecked(True)
        
        # Emit change signal (not blocked)
        self.selection_changed.emit(self.selected_ports.copy())
    
    def get_selected_ports(self):
        """Get currently selected port tuples.
        
        Returns:
            set: Copy of selected (mode, i, j) tuples
        """
        return self.selected_ports.copy()
    
    def clear_selection(self):
        """Clear all selected ports."""
        self.set_selected_ports([])


# %% [markdown]
# ## Test DifferentialPortSelector Widget

# %%
# Create and show differential selector
diff_selector = DifferentialPortSelector()
diff_selector.setWindowTitle("4-Port Differential Parameter Selector")
diff_selector.show()

# Example: Select differential parameters
# diff_selector.set_selected_ports([('DD', 1, 1), ('DD', 2, 2)])

# Example: Select mode conversion parameters to check EMI
# diff_selector.set_selected_ports([('DC', 1, 1), ('DC', 2, 1), ('CD', 1, 2)])

# %% [markdown]
# ## Usage Examples

# %% [markdown]
# ### Connect to Signals
#
# Both widgets emit signals that can be connected to plotting or analysis functions.

# %%
# Example signal connections
def on_port_selected(port_tuple):
    """Callback when a port is selected."""
    print(f"Signal received - Port selected: {port_tuple}")

def on_selection_changed(selected_set):
    """Callback when selection changes."""
    print(f"Signal received - Total selected: {len(selected_set)}")

# Connect signals
# selector.port_selected.connect(on_port_selected)
# selector.selection_changed.connect(on_selection_changed)

# %% [markdown]
# ### Programmatic Control
#
# Widgets can be controlled programmatically to sync with plot state.

# %%
# Example: Set selection from external state
# plot_traces = [(0, 0), (1, 1), (2, 2)]  # Diagonal elements
# selector.set_selected_ports(plot_traces)

# Example: Get current selection
# current = selector.get_selected_ports()
# print(f"Currently selected: {current}")

# Example: Clear all
# selector.clear_selection()

# %%