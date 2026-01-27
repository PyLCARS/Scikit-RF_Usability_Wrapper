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
# # Enum Combo Box Widgets
#
# Smart combo boxes for RF parameter enums with flexible text input.
#
# ## Features
# - **Dropdown selection** from enum values with display names
# - **Text input** leveraging enum `_missing_` for alias handling
# - **Detailed error dialogs** showing all valid inputs including aliases
# - **Auto-validation** with user-friendly messages
#
# ## Widgets
# - **EnumComboBox**: Base class for any enum type
# - **ParameterTypeSelector**: For S, Z, Y, A, T, H, G parameters
# - **PresentationTypeSelector**: For mag, db, phase, etc. (with categories)
# - **MixedModeTypeSelector**: For DD, DC, CD, CC modes
#
# Initial Author: GProtoZeroW - Jan 2026

# %% [markdown]
# ## Setup Qt Event Loop

# %%
# %gui qt

# %% [markdown]
# ## Path Setup and Imports

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
from core.scikit_rf_enums import (
    ParameterType, 
    PresentationType, 
    MixedModeType,
    ParameterTypeError,
    PresentationTypeError,
    MixedModeTypeError
)

# Get or create QApplication
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

print(f"✓ Project root: {project_root}")
print(f"✓ Imports successful")

# %% [markdown]
# ## EnumComboBox - Base Class
#
# Generic combo box that works with any enum. Features:
# - Editable for text input with alias support
# - Dropdown with display names
# - Auto-validation using enum's `_missing_` method
# - Detailed error dialogs showing all valid inputs

# %%
class EnumComboBox(QtWidgets.QWidget, WidgetUtilities):
    """Editable combo box for enum selection with alias support.
    
    Combines dropdown selection with text input, using the enum's built-in
    _missing_ method to handle aliases and case-insensitive lookup.
    
    Signals:
        value_changed(Enum): Emitted when valid enum value is selected/entered
        validation_error(str): Emitted when invalid text is entered
    
    Examples:
        >>> combo = EnumComboBox(ParameterType)
        >>> combo.value_changed.connect(lambda e: print(f"Selected: {e}"))
        >>> combo.show()
    """
    
    value_changed = QtCore.Signal(object)     # Enum member
    validation_error = QtCore.Signal(str)     # Error message
    
    def __init__(self, enum_class, parent=None):
        """Initialize with an enum class.
        
        Args:
            enum_class: The enum class (e.g., ParameterType, PresentationType)
            parent: Optional parent widget
        """
        self.enum_class = enum_class
        self._current_value = None
        self._updating = False  # Flag to prevent validation during programmatic updates
        
        super().__init__(parent)
        self._widget_startup_init_calls()
    
    def gui_layout_init(self):
        """Create combo box with label."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label showing what this selector is for
        self.label = QtWidgets.QLabel(f"{self.enum_class.__name__}:")
        
        # Editable combo box
        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        
        # Populate with enum members using display names
        for member in self.enum_class:
            display_name = member.display_name if hasattr(member, 'display_name') else str(member)
            self.combo.addItem(display_name, member)
        
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
    
    def gui_sizing_init(self):
        """Set sizing constraints."""
        self.combo.setMinimumWidth(150)
    
    def gui_styling_init(self):
        """Apply styling."""
        self.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #333;
            }
            QComboBox {
                padding: 5px;
                border: 2px solid #ccc;
                border-radius: 3px;
            }
            QComboBox:focus {
                border-color: #4CAF50;
            }
        """)
    
    def gui_wiring_init(self):
        """Connect signals."""
        # When item selected from dropdown
        self.combo.activated.connect(self.on_item_activated)
        
        # When text is edited and enter pressed or focus lost
        self.combo.lineEdit().editingFinished.connect(self.on_text_entered)
    
    def on_item_activated(self, index):
        """Handle dropdown selection (user clicked an item).
        
        Args:
            index: Index of selected item
        """
        if index >= 0:
            enum_value = self.combo.itemData(index)
            if enum_value is not None:
                self._current_value = enum_value
                self.reset_error_state()
                self.value_changed.emit(enum_value)
    
    def on_text_entered(self):
        """Handle text input using enum's _missing_ method."""
        # Skip if we're updating programmatically
        if self._updating:
            return
        
        text = self.combo.currentText().strip()
        
        if not text:
            return
        
        # Check if current text matches the display name of current selection
        # (This happens when user clicks dropdown item - don't re-validate)
        current_index = self.combo.currentIndex()
        if current_index >= 0:
            current_display = self.combo.itemText(current_index)
            if text == current_display:
                return  # Text matches dropdown selection, no need to validate
        
        try:
            # Try to convert text to enum using _missing_ method
            enum_value = self.enum_class(text)
            
            # Valid! Update combo to show the proper display name
            self._updating = True
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == enum_value:
                    self.combo.setCurrentIndex(i)
                    break
            self._updating = False
            
            self._current_value = enum_value
            self.reset_error_state()
            self.value_changed.emit(enum_value)
            
        except (ValueError, KeyError) as e:
            # Invalid input - show error dialog
            self.show_error_state()
            self.show_invalid_input_dialog(text)
            self.validation_error.emit(str(e))
            
            # Revert to last valid value
            if self._current_value is not None:
                self._updating = True
                for i in range(self.combo.count()):
                    if self.combo.itemData(i) == self._current_value:
                        self.combo.setCurrentIndex(i)
                        break
                self._updating = False
            self.reset_error_state()
    
    def show_invalid_input_dialog(self, invalid_text):
        """Show dialog explaining the invalid input and listing valid options.
        
        Args:
            invalid_text: The text that was rejected
        """
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        msg_box.setWindowTitle("Invalid Input")
        
        # Main message
        msg_box.setText(f"'{invalid_text}' is not a valid {self.enum_class.__name__}.")
        msg_box.setInformativeText("See details for all valid inputs including aliases.")
        
        # Organize valid inputs by enum member for better readability
        details_parts = ["VALID INPUTS (including all aliases):\n"]
        details_parts.append("=" * 50 + "\n")
        
        for member in self.enum_class:
            # Get display name
            display = member.display_name if hasattr(member, 'display_name') else str(member)
            
            # Get aliases for this member
            member_aliases = [alias for alias, canonical in self.enum_class._ALIASES.items() 
                             if canonical == member.name]
            
            # Combine: name, value, and aliases
            all_options = [member.name, member.value] + member_aliases
            # Remove duplicates and sort
            all_options = sorted(set(all_options), key=lambda x: (len(x), x))
            
            details_parts.append(f"\n{display}:\n")
            
            # Show in groups of 5 per line for readability
            for i in range(0, len(all_options), 5):
                line_items = all_options[i:i+5]
                details_parts.append(f"  {', '.join(line_items)}\n")
        
        details_parts.append("\n" + "=" * 50)
        details_parts.append(f"\nTotal valid inputs: {len(self.enum_class.valid_inputs())}")
        
        msg_box.setDetailedText(''.join(details_parts))
        
        # Make dialog larger and show details by default
        msg_box.setStyleSheet("QMessageBox { min-width: 400px; }")
        
        # Show dialog
        msg_box.exec()
    
    def show_error_state(self):
        """Highlight combo box as having an error."""
        self.combo.setStyleSheet("""
            QComboBox {
                border: 2px solid #f44336;
                background-color: #ffebee;
            }
        """)
    
    def reset_error_state(self):
        """Remove error highlighting."""
        self.combo.setStyleSheet("")
    
    def get_value(self):
        """Get currently selected enum value.
        
        Returns:
            Enum member or None if invalid
        """
        return self._current_value
    
    def set_value(self, enum_value):
        """Set the selected value programmatically.
        
        Args:
            enum_value: Enum member to select
        """
        self._updating = True
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == enum_value:
                self.combo.setCurrentIndex(i)
                self._current_value = enum_value
                self.reset_error_state()
                break
        self._updating = False


# %% [markdown]
# ## Test EnumComboBox with ParameterType

# %%
# Test with ParameterType enum
param_combo = EnumComboBox(ParameterType)
param_combo.setWindowTitle("Parameter Type Selector")
param_combo.show()

# Connect to see output
param_combo.value_changed.connect(lambda e: print(f"✓ Valid: {e} ({e.display_name})"))
param_combo.validation_error.connect(lambda msg: print(f"✗ Error detected"))

# %% [markdown]
# ### Try These Inputs
#
# **Valid inputs** (should work):
# - `s` → S-Parameters
# - `scattering` → S-Parameters
# - `impedance` → Z-Parameters
# - `ABCD` → A-Parameters
#
# **Invalid input** (should show error dialog):
# - `cat` → Shows dialog with all valid options
# - `xyz` → Shows dialog with all valid options

# %% [markdown]
# ## ParameterTypeSelector - Specialized Widget
#
# Wrapper around EnumComboBox specifically for ParameterType with better labeling.

# %%
class ParameterTypeSelector(EnumComboBox):
    """Specialized selector for network parameter types (S, Z, Y, etc.).
    
    Examples:
        >>> selector = ParameterTypeSelector()
        >>> selector.value_changed.connect(on_parameter_changed)
        >>> selector.set_value(ParameterType.S)
    """
    
    def __init__(self, parent=None):
        """Initialize parameter type selector."""
        super().__init__(ParameterType, parent)
    
    def gui_layout_init(self):
        """Override to customize label."""
        super().gui_layout_init()
        self.label.setText("Parameter Representation:")


# %% [markdown]
# ## Test ParameterTypeSelector

# %%
param_selector = ParameterTypeSelector()
param_selector.setWindowTitle("Parameter Type Selector")
param_selector.show()

# Set initial value
param_selector.set_value(ParameterType.S)

param_selector.value_changed.connect(lambda e: print(f"Parameter changed to: {e.display_name}"))

# %% [markdown]
# ## PresentationTypeSelector - With Category Separators
#
# Enhanced selector that groups presentation types by category.

# %%
class PresentationTypeSelector(QtWidgets.QWidget, WidgetUtilities):
    """Selector for presentation types with category organization.
    
    Groups presentations by category (magnitude, phase, complex, time, special)
    for easier navigation.
    
    Signals:
        value_changed(PresentationType): Emitted when selection changes
        validation_error(str): Emitted on invalid text input
    """
    
    value_changed = QtCore.Signal(object)
    validation_error = QtCore.Signal(str)
    
    def __init__(self, parent=None):
        """Initialize presentation type selector."""
        self._current_value = None
        self._updating = False
        
        super().__init__(parent)
        self._widget_startup_init_calls()
    
    def gui_layout_init(self):
        """Create combo with category separators."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QtWidgets.QLabel("Presentation:")
        
        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        
        # Populate by category
        categories = PresentationType._CATEGORIES
        for category_name, member_names in categories.items():
            # Add category separator
            self.combo.addItem(f"─── {category_name.upper()} ───")
            separator_idx = self.combo.count() - 1
            self.combo.model().item(separator_idx).setEnabled(False)
            self.combo.setItemData(separator_idx, None)  # No data for separators
            
            # Add members in this category
            for member_name in member_names:
                member = PresentationType[member_name]
                self.combo.addItem(member.display_name, member)
        
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
    
    def gui_sizing_init(self):
        """Set sizing."""
        self.combo.setMinimumWidth(200)
    
    def gui_styling_init(self):
        """Apply styling."""
        self.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #333;
            }
            QComboBox {
                padding: 5px;
                border: 2px solid #ccc;
                border-radius: 3px;
            }
            QComboBox:focus {
                border-color: #4CAF50;
            }
        """)
    
    def gui_wiring_init(self):
        """Connect signals."""
        self.combo.activated.connect(self.on_item_activated)
        self.combo.lineEdit().editingFinished.connect(self.on_text_entered)
    
    def on_item_activated(self, index):
        """Handle dropdown selection."""
        if index >= 0:
            enum_value = self.combo.itemData(index)
            if enum_value is not None:  # Skip separators
                self._current_value = enum_value
                self.combo.setStyleSheet("")
                self.value_changed.emit(enum_value)
    
    def on_text_entered(self):
        """Handle text input."""
        if self._updating:
            return
        
        text = self.combo.currentText().strip()
        
        if not text:
            return
        
        # Check if matches current display name
        current_index = self.combo.currentIndex()
        if current_index >= 0:
            current_display = self.combo.itemText(current_index)
            if text == current_display:
                return
        
        try:
            enum_value = PresentationType(text)
            
            # Update to show proper display name
            self._updating = True
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == enum_value:
                    self.combo.setCurrentIndex(i)
                    break
            self._updating = False
            
            self._current_value = enum_value
            self.combo.setStyleSheet("")
            self.value_changed.emit(enum_value)
            
        except (ValueError, KeyError) as e:
            self.combo.setStyleSheet("QComboBox { border: 2px solid #f44336; background-color: #ffebee; }")
            self.show_invalid_input_dialog(text)
            self.validation_error.emit(str(e))
            
            # Revert to last valid
            if self._current_value is not None:
                self._updating = True
                for i in range(self.combo.count()):
                    if self.combo.itemData(i) == self._current_value:
                        self.combo.setCurrentIndex(i)
                        break
                self._updating = False
            self.combo.setStyleSheet("")
    
    def show_invalid_input_dialog(self, invalid_text):
        """Show dialog with all valid PresentationType inputs."""
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        msg_box.setWindowTitle("Invalid Input")
        msg_box.setText(f"'{invalid_text}' is not a valid PresentationType.")
        msg_box.setInformativeText("See details for all valid inputs organized by category.")
        
        # Organize by category
        details_parts = ["VALID INPUTS BY CATEGORY:\n"]
        details_parts.append("=" * 60 + "\n")
        
        for category, member_names in PresentationType._CATEGORIES.items():
            details_parts.append(f"\n{category.upper()}:\n")
            details_parts.append("-" * 60 + "\n")
            
            for member_name in member_names:
                member = PresentationType[member_name]
                
                # Get aliases
                member_aliases = [alias for alias, canonical in PresentationType._ALIASES.items() 
                                if canonical == member.name]
                
                all_options = [member.name, member.value] + member_aliases
                all_options = sorted(set(all_options), key=lambda x: (len(x), x))
                
                details_parts.append(f"\n  {member.display_name}:\n")
                
                # Show in groups of 5
                for i in range(0, len(all_options), 5):
                    line_items = all_options[i:i+5]
                    details_parts.append(f"    {', '.join(line_items)}\n")
        
        details_parts.append("\n" + "=" * 60)
        details_parts.append(f"\nTotal valid inputs: {len(PresentationType.valid_inputs())}")
        
        msg_box.setDetailedText(''.join(details_parts))
        msg_box.setStyleSheet("QMessageBox { min-width: 500px; }")
        msg_box.exec()
    
    def get_value(self):
        """Get current value."""
        return self._current_value
    
    def set_value(self, enum_value):
        """Set value programmatically."""
        self._updating = True
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == enum_value:
                self.combo.setCurrentIndex(i)
                self._current_value = enum_value
                break
        self._updating = False


# %% [markdown]
# ## Test PresentationTypeSelector

# %%
pres_selector = PresentationTypeSelector()
pres_selector.setWindowTitle("Presentation Type Selector")
pres_selector.show()

pres_selector.value_changed.connect(lambda e: print(f"✓ {e.display_name} (category: {e.category})"))

# Set initial value
pres_selector.set_value(PresentationType.DB)

# %% [markdown]
# ### Try These Inputs
#
# **Valid:**
# - `mag` → Magnitude
# - `decibel` → Magnitude (dB)
# - `phase` → Phase (degrees)
# - `smith` → Smith Chart
# - `real` → Real Part
#
# **Invalid:**
# - `cat` → Shows organized list by category

# %% [markdown]
# ## MixedModeTypeSelector
#
# Simple selector for 4-port differential modes.

# %%
class MixedModeTypeSelector(EnumComboBox):
    """Selector for mixed-mode types (DD, DC, CD, CC).
    
    Examples:
        >>> selector = MixedModeTypeSelector()
        >>> selector.set_value(MixedModeType.DD)
    """
    
    def __init__(self, parent=None):
        """Initialize mixed-mode selector."""
        super().__init__(MixedModeType, parent)
    
    def gui_layout_init(self):
        """Override to customize label."""
        super().gui_layout_init()
        self.label.setText("Mixed-Mode Type:")


# %% [markdown]
# ## Test MixedModeTypeSelector

# %%
mode_selector = MixedModeTypeSelector()
mode_selector.setWindowTitle("Mixed-Mode Type Selector")
mode_selector.show()

mode_selector.value_changed.connect(
    lambda e: print(f"✓ {e.display_name} - Mode conversion: {e.is_mode_conversion}")
)

mode_selector.set_value(MixedModeType.DD)

# %% [markdown]
# ## Combined Example - Parameter Selection Panel
#
# Combine all three selectors in a single panel.

# %%
class ParameterSelectionPanel(QtWidgets.QWidget, WidgetUtilities):
    """Combined panel with parameter, presentation, and mode selectors.
    
    Signals:
        selection_changed(dict): Emitted when any selector changes
    """
    
    selection_changed = QtCore.Signal(dict)
    
    def __init__(self, parent=None):
        """Initialize selection panel."""
        super().__init__(parent)
        self._widget_startup_init_calls()
    
    def gui_layout_init(self):
        """Create layout with all three selectors."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("RF Parameter Selection")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Selectors
        self.param_selector = ParameterTypeSelector()
        self.pres_selector = PresentationTypeSelector()
        self.mode_selector = MixedModeTypeSelector()
        
        layout.addWidget(self.param_selector)
        layout.addWidget(self.pres_selector)
        layout.addWidget(self.mode_selector)
        
        # Status display
        self.status_label = QtWidgets.QLabel("Make selections above")
        self.status_label.setStyleSheet("""
            padding: 10px;
            background-color: #f5f5f5;
            border: 1px solid #ccc;
            border-radius: 3px;
            font-family: monospace;
        """)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def gui_sizing_init(self):
        """Set sizing."""
        self.setMinimumWidth(350)
        self.setMinimumHeight(400)
    
    def gui_styling_init(self):
        """Apply styling."""
        pass
    
    def gui_wiring_init(self):
        """Connect all selectors."""
        self.param_selector.value_changed.connect(self.on_selection_changed)
        self.pres_selector.value_changed.connect(self.on_selection_changed)
        self.mode_selector.value_changed.connect(self.on_selection_changed)
    
    def on_selection_changed(self):
        """Handle any selection change."""
        selection = {
            'parameter': self.param_selector.get_value(),
            'presentation': self.pres_selector.get_value(),
            'mode': self.mode_selector.get_value(),
        }
        
        # Update status display
        status_parts = ["Current Selection:"]
        status_parts.append("=" * 40)
        
        if selection['parameter']:
            status_parts.append(f"Parameter:    {selection['parameter'].display_name}")
        else:
            status_parts.append("Parameter:    (none)")
            
        if selection['presentation']:
            status_parts.append(f"Presentation: {selection['presentation'].display_name}")
        else:
            status_parts.append("Presentation: (none)")
            
        if selection['mode']:
            mode_conv = " [Mode Conversion]" if selection['mode'].is_mode_conversion else ""
            status_parts.append(f"Mode:         {selection['mode'].display_name}{mode_conv}")
        else:
            status_parts.append("Mode:         (none)")
        
        status_text = "\n".join(status_parts)
        self.status_label.setText(status_text)
        
        self.selection_changed.emit(selection)
    
    def get_selection(self):
        """Get current selection state.
        
        Returns:
            dict with 'parameter', 'presentation', 'mode' keys
        """
        return {
            'parameter': self.param_selector.get_value(),
            'presentation': self.pres_selector.get_value(),
            'mode': self.mode_selector.get_value(),
        }


# %% [markdown]
# ## Test Combined Panel

# %%
panel = ParameterSelectionPanel()
panel.setWindowTitle("RF Parameter Selection Panel")
panel.show()

# Connect to see changes
panel.selection_changed.connect(lambda s: print(f"Selection updated: {list(s.keys())}"))

# Set some defaults
panel.param_selector.set_value(ParameterType.S)
panel.pres_selector.set_value(PresentationType.DB)
panel.mode_selector.set_value(MixedModeType.DD)

# %% [markdown]
# ## Alias Testing
#
# Test the flexible text input with various aliases.

# %%
# Test different alias formats
print("Testing ParameterType aliases:")
print("=" * 50)
test_inputs = ['s', 'S', 'scattering', 'S-PARAMETERS', 'sparams', 'impedance', 'z', 'ABCD']

for text in test_inputs:
    try:
        result = ParameterType(text)
        print(f"  ✓ '{text:15}' → {result.name} ({result.display_name})")
    except (ValueError, KeyError) as e:
        print(f"  ✗ '{text:15}' → INVALID")

# %%
print("\nTesting PresentationType aliases:")
print("=" * 50)
test_inputs = ['mag', 'MAGNITUDE', 'db', 'decibel', 'phase', 'deg', 'smith', 'vswr', 're', 'imaginary']

for text in test_inputs:
    try:
        result = PresentationType(text)
        print(f"  ✓ '{text:15}' → {result.name:10} ({result.display_name}) [{result.category}]")
    except (ValueError, KeyError) as e:
        print(f"  ✗ '{text:15}' → INVALID")

# %%
print("\nTesting MixedModeType aliases:")
print("=" * 50)
test_inputs = ['dd', 'DD', 'diff-diff', 'dc', 'differential-common', 'cd', 'cc', 'common-common']

for text in test_inputs:
    try:
        result = MixedModeType(text)
        conv = " [MODE CONVERSION]" if result.is_mode_conversion else ""
        print(f"  ✓ '{text:20}' → {result.name} ({result.display_name}){conv}")
    except (ValueError, KeyError) as e:
        print(f"  ✗ '{text:20}' → INVALID")

# %%
