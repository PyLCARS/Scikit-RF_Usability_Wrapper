# %% [markdown]
# # PySide6 Widget Utilities Base Class
#
# Base class providing standard patterns for PySide6 widget development with
# consistent initialization, debug tools, and development utilities.
#
# ## Key Features
# - **Standard Init Pattern**: Enforces consistent widget setup sequence
# - **Signal Blocking**: Context manager for temporary signal blocking
# - **Debug Tools**: Tooltips and widget capture for automation/testing
#
# ## Initial Author
# GProtoZeroW - Sept 2025
#
# ## Acknowledgments
# Documentation and code comments refined with assistance from Claude Opus 4
# (Anthropic) for proofreading, spelling corrections, and clarity improvements.

# %%
from PySide6 import QtWidgets, QtCore, QtGui
from contextlib import contextmanager
import base64
import sys


# %% [markdown]
# ## WidgetUtilities Base Class
#
# ### Purpose
# This class serves three main purposes:
#
# **Part 1: Enforcing Widget Development Standards**
#
# Provides a standard set of methods to override when creating new widget classes,
# with automatic initialization sequence execution.
#
# Methods to Override:
# - `gui_layout_init()`: Instantiate child widgets and set up layouts
# - `gui_sizing_init()`: Set sizing constraints (min/max sizes, etc.)
# - `gui_styling_init()`: Apply custom styling
# - `gui_wiring_init()`: Connect signals to slots
#
# Initialization Sequence (via `_widget_startup_init_calls()`):
# 1. Layout → 2. Sizing → 3. Styling → 4. Wiring
#
# This order minimizes unintended signal firing during setup.
#
# **Part 2: Development Utilities**
# - `temp_block_widgets_signals()`: Context manager for signal blocking
#
# **Part 3: Debug and Automation Utilities**
# - `set_debug_tooltips()`: Add widget name/type to tooltips
# - `capture_widget_info()`: Capture widget screenshots and positions
# - `show_widget_image()`: Display captured widget images
#
# ### Note on ABC Implementation
# While conceptually an "Abstract Base Class", this doesn't use Python's `abc`
# module due to metaclass conflicts with Qt. Instead uses `raise NotImplementedError` -
# simpler and equally effective for runtime-called methods.

# %%
class WidgetUtilities:
    """Base class providing common widget utilities and initialization structure.
    
    Provides debug tools for widget inspection and a standard initialization
    sequence for GUI widgets. Subclasses must implement the abstract
    initialization methods.
    
    Attributes:
        None (base class only defines methods)
    
    Examples:
        >>> class MyWidget(QtWidgets.QWidget, WidgetUtilities):
        ...     def __init__(self, parent=None):
        ...         super().__init__(parent)
        ...         self._widget_startup_init_calls()
        ...     
        ...     def gui_layout_init(self):
        ...         # Create and arrange widgets
        ...         self.button = QtWidgets.QPushButton("Click Me")
        ...         layout = QtWidgets.QVBoxLayout(self)
        ...         layout.addWidget(self.button)
        ...     
        ...     def gui_sizing_init(self):
        ...         self.setMinimumSize(200, 100)
        ...     
        ...     def gui_styling_init(self):
        ...         self.setStyleSheet("background-color: white;")
        ...     
        ...     def gui_wiring_init(self):
        ...         self.button.clicked.connect(self.on_click)
    
    Notes:
        - Call `_widget_startup_init_calls()` in your `__init__` after `super().__init__()`
        - All four init methods must be implemented (even if empty)
        - Wiring comes last to avoid signals during setup
    """

    # ===========================================
    # === Standard Widget Initialization API ===
    # ===========================================

    def _widget_startup_init_calls(self):
        """Execute all GUI initialization methods in proper sequence.
        
        Call this method in your widget's `__init__` after calling `super().__init__()`.
        It will execute the initialization methods in the recommended order:
        Layout → Sizing → Styling → Wiring
        
        Examples:
            >>> def __init__(self, parent=None):
            ...     super().__init__(parent)
            ...     self._widget_startup_init_calls()  # Must call this
        """
        self.gui_layout_init()
        self.gui_sizing_init()
        self.gui_styling_init()
        self.gui_wiring_init()

    def gui_layout_init(self):
        """Initialize and arrange all GUI elements.
        
        Create all child widgets and set up the layout hierarchy. This is the
        first method called in the initialization sequence.
        
        Raises:
            NotImplementedError: Must be implemented by subclass
            
        Examples:
            >>> def gui_layout_init(self):
            ...     self.label = QtWidgets.QLabel("Name:")
            ...     self.line_edit = QtWidgets.QLineEdit()
            ...     
            ...     layout = QtWidgets.QHBoxLayout(self)
            ...     layout.addWidget(self.label)
            ...     layout.addWidget(self.line_edit)
        """
        raise NotImplementedError("Subclass must implement gui_layout_init")

    def gui_sizing_init(self):
        """Set widget size constraints and properties.
        
        Configure minimum/maximum sizes, fixed dimensions, size policies, etc.
        Called after layout initialization.
        
        Raises:
            NotImplementedError: Must be implemented by subclass
            
        Examples:
            >>> def gui_sizing_init(self):
            ...     self.setMinimumSize(300, 200)
            ...     self.setMaximumHeight(400)
            ...     self.line_edit.setFixedWidth(200)
        """
        raise NotImplementedError("Subclass must implement gui_sizing_init")

    def gui_styling_init(self):
        """Apply widget styling and appearance settings.
        
        Set stylesheets, fonts, colors, icons, etc. Called after sizing to
        ensure style calculations have correct dimensions.
        
        Raises:
            NotImplementedError: Must be implemented by subclass
            
        Examples:
            >>> def gui_styling_init(self):
            ...     self.setStyleSheet("QLabel { color: blue; }")
            ...     self.button.setIcon(QtGui.QIcon("icon.png"))
        """
        raise NotImplementedError("Subclass must implement gui_styling_init")

    def gui_wiring_init(self):
        """Connect widget signals to their slots.
        
        Set up all signal-slot connections. Called last to avoid firing signals
        during widget construction (e.g., resize events before styling complete).
        
        Raises:
            NotImplementedError: Must be implemented by subclass
            
        Examples:
            >>> def gui_wiring_init(self):
            ...     self.button.clicked.connect(self.on_button_clicked)
            ...     self.line_edit.textChanged.connect(self.on_text_changed)
        """
        raise NotImplementedError("Subclass must implement gui_wiring_init")

    # ========================
    # === Usage Utilities ===
    # ========================

    @contextmanager
    def temp_block_widgets_signals(self, *widgets):
        """Temporarily block signals for multiple widgets.
        
        Context manager that blocks all signals from specified widgets for the
        duration of the context, then restores their previous signal state.
        Essential for preventing circular signal emission when programmatically
        updating interconnected widget values.
        
        Args:
            *widgets: Variable number of QWidget instances to block
            
        Yields:
            None
            
        Examples:
            >>> with self.temp_block_widgets_signals(self.lineEdit, self.comboBox):
            ...     self.lineEdit.setText("new value")
            ...     self.comboBox.setCurrentIndex(2)
            ...     # No signals emitted from these widgets during this block
        
        Notes:
            - If a widget was already blocking signals before entering, it
              will remain blocked after exiting
            - Non-QObject items are skipped with a warning
            - Safe to nest multiple context managers
        """
        # Store original signal blocking states
        original_states = []
        
        for widget in widgets:
            if isinstance(widget, QtCore.QObject):  # Check it's a Qt object
                original_states.append((widget, widget.blockSignals(True)))
            else:
                # Use print instead of logger for base class (user can add logging)
                print(f"Warning: Skipping non-QObject: {widget}")
                original_states.append((widget, None))
        
        try:
            yield
        finally:
            # Restore original signal blocking states
            for widget, original_state in original_states:
                if original_state is not None:  # Was a valid QObject
                    widget.blockSignals(original_state)

    # ===================
    # === Debug Tools ===
    # ===================

    def set_debug_tooltips(self):
        """Add debug information to all widget tooltips.
        
        Iterates through all QWidget attributes and adds (or appends) debug
        information showing the widget's attribute name and type. Invaluable
        for identifying widgets in the rendered UI during development.
        
        If a widget already has a tooltip, the debug info is appended.
        Otherwise, the debug tooltip is set directly.
        
        Examples:
            >>> widget = MyWidget()
            >>> widget.set_debug_tooltips()
            >>> # Now hovering shows: "Widget: my_button\nType: QPushButton"
        
        Notes:
            - Call after all widgets are created (e.g., end of __init__)
            - Also adds tooltip to the parent widget itself
            - Safe to call multiple times (will update tooltips)
        """
        # Add tooltips to all child widgets
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, QtWidgets.QWidget):
                debug_info = f"Widget: {attr_name}\nType: {type(attr).__name__}"
                
                existing_tooltip = attr.toolTip()
                if existing_tooltip:
                    # Append debug info to existing tooltip
                    attr.setToolTip(f"{existing_tooltip}\n\n--- Debug Info ---\n{debug_info}")
                else:
                    # No existing tooltip, just set debug info
                    attr.setToolTip(debug_info)
        
        # Add tooltip to parent widget itself
        if isinstance(self, QtWidgets.QWidget):
            debug_info = f"Widget: {self.__class__.__name__}"
            existing_tooltip = self.toolTip()
            
            if existing_tooltip:
                self.setToolTip(f"{existing_tooltip}\n\n--- Debug Info ---\n{debug_info}")
            else:
                self.setToolTip(debug_info)

    def capture_widget_info(self):
        """Capture widget images and locations for automation/testing.
        
        Iterates through all visible QWidget attributes and captures their
        screen position, size, and rendered appearance as base64-encoded PNG.
        Intended for use with pyautogui or similar automation tools for
        image-based widget location.
        
        Returns:
            dict: Widget information indexed by attribute name. Each entry contains:
                - location (tuple[int, int]): (x, y) global screen coordinates
                  of the widget's top-left corner
                - size (tuple[int, int]): (width, height) in pixels
                - image_base64 (str): Base64 encoded PNG image of widget
                - type (str): Widget class name (e.g., 'QPushButton')
                
        Examples:
            >>> info = widget.capture_widget_info()
            >>> info['self']['type']  # Parent widget itself
            'MyCustomWidget'
            >>> info['submit_button']['location']
            (100, 200)
            >>> # Use with pyautogui:
            >>> from PIL import Image
            >>> from io import BytesIO
            >>> img_data = base64.b64decode(info['submit_button']['image_base64'])
            >>> img = Image.open(BytesIO(img_data))
            >>> # img.save('button.png')  # Or use with pyautogui.locateOnScreen()
        
        Notes:
            - Only captures visible widgets
            - Coordinates are global screen coordinates
            - 'self' key contains parent widget info
            - Images are PNG format in base64 encoding
        """
        widget_info = {}
        
        # First, capture the parent widget itself if it's a QWidget
        if isinstance(self, QtWidgets.QWidget) and self.isVisible():
            # Get screen position of widget's top-left corner
            global_pos = self.mapToGlobal(QtCore.QPoint(0, 0))
            location = (global_pos.x(), global_pos.y())
            size = (self.width(), self.height())
            
            # Create a pixmap (off-screen image) of the widget's size
            pixmap = QtGui.QPixmap(self.size())
            # Render the widget's visual appearance into the pixmap
            self.render(pixmap)
            
            # Convert pixmap to base64 string for storage/transmission
            byte_array = QtCore.QByteArray()
            buffer = QtCore.QBuffer(byte_array)
            buffer.open(QtCore.QIODevice.WriteOnly)
            pixmap.save(buffer, "PNG")  # Save pixmap as PNG into buffer
            base64_str = base64.b64encode(byte_array.data()).decode()
            
            widget_info['self'] = {
                'location': location,
                'size': size,
                'image_base64': base64_str,
                'type': type(self).__name__
            }
        
        # Then capture all child widgets
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, QtWidgets.QWidget) and attr.isVisible():
                # Get widget's position in global screen coordinates
                global_pos = attr.mapToGlobal(QtCore.QPoint(0, 0))
                location = (global_pos.x(), global_pos.y())
                size = (attr.width(), attr.height())
                
                # Create pixmap and render widget appearance
                pixmap = QtGui.QPixmap(attr.size())
                attr.render(pixmap)
                
                # Convert to base64 for easy storage/transmission
                byte_array = QtCore.QByteArray()
                buffer = QtCore.QBuffer(byte_array)
                buffer.open(QtCore.QIODevice.WriteOnly)
                pixmap.save(buffer, "PNG")
                base64_str = base64.b64encode(byte_array.data()).decode()
                
                widget_info[attr_name] = {
                    'location': location,
                    'size': size,
                    'image_base64': base64_str,
                    'type': type(attr).__name__
                }
        
        return widget_info

    @staticmethod
    def show_widget_image(widget_info, widget_name):
        """Display captured widget image with formatted metadata.
        
        Static method to visualize widget screenshots captured by
        `capture_widget_info()`. Uses matplotlib to display the image
        with location and size information.
        
        Args:
            widget_info (dict): Dictionary returned from capture_widget_info()
            widget_name (str): Key of the widget to display (e.g., 'submit_button')
        
        Raises:
            KeyError: If widget_name not found in widget_info
            ImportError: If matplotlib or PIL not available
            
        Examples:
            >>> info = widget.capture_widget_info()
            >>> WidgetUtilities.show_widget_image(info, 'self')
            >>> WidgetUtilities.show_widget_image(info, 'my_button')
        
        Notes:
            - Requires matplotlib and PIL (Pillow) to be installed
            - Best used in Jupyter notebooks or interactive sessions
            - Image shows actual rendered appearance of widget
        """
        # Import here to avoid requiring matplotlib for all users
        try:
            from PIL import Image
            from io import BytesIO
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "show_widget_image requires matplotlib and Pillow. "
                "Install with: pip install matplotlib pillow"
            ) from e
        
        # Decode base64 image
        img_data = base64.b64decode(widget_info[widget_name]['image_base64'])
        img = Image.open(BytesIO(img_data))
        
        # Create figure with better spacing
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Add border around image
        ax.add_patch(plt.Rectangle((0, 0), img.width-1, img.height-1,
                                  fill=False, edgecolor='gray', linewidth=2))
        
        # Format widget info
        info = widget_info[widget_name]
        
        # Main title - widget name and type
        ax.set_title(f"{widget_name} ({info['type']})",
                    fontsize=14, fontweight='bold', pad=10)
        
        # Subtitle with location and size info
        location_text = (f"Widget Top-Left corner location in screen space: "
                        f"{info['location']} • Size: {info['size'][0]}×{info['size'][1]}px")
        fig.text(0.5, 0.92, location_text,
                ha='center', fontsize=10, color='gray')
        
        # Tighten layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        plt.show()


# %% [markdown]
# ## Helper Functions
#
# Utility functions for widget development and testing.

# %%
def close_qt_windows():
    """Close all visible top-level Qt windows.
    
    Iterates through all top-level widgets in the application and closes
    any that are currently visible. Useful for cleaning up during
    interactive development in notebooks.
    
    Returns:
        int: Number of windows closed
        
    Examples:
        >>> close_qt_windows()  # Clean up before creating new widgets
        2
    
    Notes:
        - Safe to call when no windows exist (returns 0)
        - Only closes visible windows (minimized windows ignored)
        - Useful in Jupyter notebooks to avoid window clutter
    """
    closed_count = 0
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        return closed_count
    
    for widget in app.topLevelWidgets():
        if widget.isVisible():
            widget.close()
            closed_count += 1
    
    return closed_count


# %% [markdown]
# ## Usage Example
#
# Demonstrates how to create a widget using WidgetUtilities as a base class.

# %%
class ExampleWidget(QtWidgets.QWidget, WidgetUtilities):
    """Example widget demonstrating WidgetUtilities usage.
    
    A simple widget with a label, line edit, and button that shows
    how to properly structure a widget using the WidgetUtilities pattern.
    """
    
    def __init__(self, parent=None):
        """Initialize the example widget.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # IMPORTANT: Call this after super().__init__()
        self._widget_startup_init_calls()
        
        # Optional: Enable debug tooltips during development
        # self.set_debug_tooltips()
    
    def gui_layout_init(self):
        """Create and arrange widgets."""
        # Create widgets
        self.label = QtWidgets.QLabel("Name:")
        self.line_edit = QtWidgets.QLineEdit()
        self.button = QtWidgets.QPushButton("Submit")
        self.status_label = QtWidgets.QLabel("")
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Add widgets to layout
        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(self.label)
        input_layout.addWidget(self.line_edit)
        
        layout.addLayout(input_layout)
        layout.addWidget(self.button)
        layout.addWidget(self.status_label)
    
    def gui_sizing_init(self):
        """Set widget sizes."""
        self.setMinimumSize(300, 150)
        self.line_edit.setMinimumWidth(200)
    
    def gui_styling_init(self):
        """Apply styling."""
        self.setStyleSheet("""
            QLabel { color: #333; }
            QPushButton { 
                background-color: #4CAF50; 
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
    
    def gui_wiring_init(self):
        """Connect signals to slots."""
        self.button.clicked.connect(self.on_submit)
        self.line_edit.returnPressed.connect(self.on_submit)
    
    def on_submit(self):
        """Handle submit button click."""
        name = self.line_edit.text()
        if name:
            self.status_label.setText(f"Hello, {name}!")
        else:
            self.status_label.setText("Please enter a name.")


# %% [markdown]
# ## Main Function
#
# Demonstrates the example widget when run as a script.

# %%
def main():
    """Run example widget demonstration."""
    import sys
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Close any existing windows
    close_qt_windows()
    
    # Create and show example widget
    widget = ExampleWidget()
    widget.setWindowTitle("WidgetUtilities Example")
    widget.show()
    
    # For notebook use, just show the widget
    # For script use, uncomment the next line:
    # sys.exit(app.exec())


# %%
if __name__ == "__main__":
    main()
