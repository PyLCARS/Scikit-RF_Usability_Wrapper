# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Touchstone Analyzer - Network Parameter Enums
#
# This module defines enums for network parameter types, presentation formats,
# and mixed-mode analysis. All enums support case-insensitive lookups and
# common aliases for user convenience.
#
# ## Features
# - **ParameterType**: Network representations (S, Z, Y, A, T, H, G parameters)
# - **PresentationType**: Data presentations (magnitude, phase, complex, time domain, etc.)
# - **MixedModeType**: 4-port mixed-mode analysis (DD, DC, CD, CC)
# - **Flexible Input**: Case-insensitive with extensive alias support
# - **Custom Exceptions**: Helpful error messages with valid options
#
# ## Quick Example
# ```python
# # All of these work:
# ParameterType('s')           # → S
# ParameterType('SCATTERING')  # → S
# ParameterType('s-params')    # → S
#
# PresentationType('mag')      # → MAG
# PresentationType('DECIBEL')  # → DB
# ```

# %%
from enum import Enum
from typing import List, Optional


# %% [markdown]
# ## ParameterType Enum
#
# Represents network parameter types (representations) used in RF analysis.
# Supports all common 2-port parameter types with extensive alias support.

# %%
class ParameterType(Enum):
    """Network parameter representation types.
    
    Supports the seven common network parameter representations used in
    RF and microwave engineering. Each representation provides different
    insights into circuit behavior.
    
    Attributes:
        S: Scattering parameters (power waves)
        Z: Impedance parameters (voltage-current, series)
        Y: Admittance parameters (current-voltage, parallel)
        A: ABCD parameters (transmission/chain)
        T: Scattering transfer parameters
        H: Hybrid parameters (mixed)
        G: Inverse hybrid parameters
    
    Examples:
        >>> ParameterType('s')
        <ParameterType.S: 's'>
        >>> ParameterType('SCATTERING')
        <ParameterType.S: 's'>
        >>> ParameterType.Z.display_name
        'Z-Parameters (Impedance)'
        >>> str(ParameterType.S)
        'S'
    
    Notes:
        - Lookups are case-insensitive
        - Supports many common aliases (see _ALIASES)
        - Use .display_name for UI-friendly strings
    """
    
    # Tell Python these aren't enum members
    _ignore_ = ['_ALIASES', '_DISPLAY_NAMES']
    
    # Enum members
    S = 's'  # Scattering parameters
    Z = 'z'  # Impedance parameters
    Y = 'y'  # Admittance parameters
    A = 'a'  # ABCD parameters
    T = 't'  # Scattering transfer parameters
    H = 'h'  # Hybrid parameters
    G = 'g'  # Inverse hybrid parameters


# %%
# Define aliases after the class to avoid enum member conflict
ParameterType._ALIASES = {
    # S-parameters
    's': 'S',
    'scattering': 'S',
    's-parameters': 'S',
    's-params': 'S',
    'sparameters': 'S',
    'sparams': 'S',
    's_parameters': 'S',
    's_params': 'S',
    
    # Z-parameters
    'z': 'Z',
    'impedance': 'Z',
    'z-parameters': 'Z',
    'z-params': 'Z',
    'zparameters': 'Z',
    'zparams': 'Z',
    'z_parameters': 'Z',
    'z_params': 'Z',
    
    # Y-parameters
    'y': 'Y',
    'admittance': 'Y',
    'y-parameters': 'Y',
    'y-params': 'Y',
    'yparameters': 'Y',
    'yparams': 'Y',
    'y_parameters': 'Y',
    'y_params': 'Y',
    
    # A-parameters (ABCD)
    'a': 'A',
    'abcd': 'A',
    'a-parameters': 'A',
    'a-params': 'A',
    'aparameters': 'A',
    'aparams': 'A',
    'a_parameters': 'A',
    'a_params': 'A',
    'transmission': 'A',
    
    # T-parameters
    't': 'T',
    'scattering-transfer': 'T',
    'scattering_transfer': 'T',
    'transfer': 'T',
    't-parameters': 'T',
    't-params': 'T',
    'tparameters': 'T',
    'tparams': 'T',
    't_parameters': 'T',
    't_params': 'T',
    
    # H-parameters
    'h': 'H',
    'hybrid': 'H',
    'h-parameters': 'H',
    'h-params': 'H',
    'hparameters': 'H',
    'hparams': 'H',
    'h_parameters': 'H',
    'h_params': 'H',
    
    # G-parameters
    'g': 'G',
    'inverse-hybrid': 'G',
    'inverse_hybrid': 'G',
    'inversehybrid': 'G',
    'g-parameters': 'G',
    'g-params': 'G',
    'gparameters': 'G',
    'gparams': 'G',
    'g_parameters': 'G',
    'g_params': 'G',
}

ParameterType._DISPLAY_NAMES = {
    'S': 'S-Parameters (Scattering)',
    'Z': 'Z-Parameters (Impedance)',
    'Y': 'Y-Parameters (Admittance)',
    'A': 'A-Parameters (ABCD)',
    'T': 'T-Parameters (Scattering Transfer)',
    'H': 'H-Parameters (Hybrid)',
    'G': 'G-Parameters (Inverse Hybrid)',
}


# %%
@classmethod
def _parameter_type_missing(cls, value):
    """Handle case-insensitive and alias lookups.
    
    This method is automatically called by Python's Enum machinery when
    a lookup fails with the standard mechanism.
    
    Args:
        value: The value to look up (typically a string)
        
    Returns:
        The matching enum member, or None if not found
        
    Examples:
        >>> ParameterType('SCATTERING')  # Calls _missing_
        <ParameterType.S: 's'>
    """
    if not isinstance(value, str):
        return None
        
    # Normalize: lowercase and strip whitespace
    normalized = value.lower().strip()
    
    # Try to find in aliases
    canonical_name = cls._ALIASES.get(normalized)
    if canonical_name:
        return cls[canonical_name]
    
    return None


def _parameter_type_display_name(self) -> str:
    """Get user-friendly display name.
    
    Returns:
        Human-readable name suitable for UI display
        
    Examples:
        >>> ParameterType.S.display_name
        'S-Parameters (Scattering)'
    """
    return self._DISPLAY_NAMES[self.name]


@classmethod
def _parameter_type_valid_inputs(cls) -> List[str]:
    """Get list of all valid input strings (canonical + aliases).
    
    Returns:
        Sorted list of all accepted input strings
        
    Examples:
        >>> 'scattering' in ParameterType.valid_inputs()
        True
    """
    # Get all aliases
    aliases = set(cls._ALIASES.keys())
    # Add canonical enum values
    canonical = {member.value for member in cls}
    # Add uppercase names
    names = {member.name for member in cls}
    
    return sorted(aliases | canonical | names)


@classmethod
def _parameter_type_valid_types(cls) -> str:
    """Get formatted string of valid parameter types for error messages.
    
    Returns:
        Comma-separated string of canonical parameter types
        
    Examples:
        >>> ParameterType.valid_types()
        'S, Z, Y, A, T, H, G'
    """
    return ', '.join([member.name for member in cls])


def _parameter_type_str(self):
    """String representation returns the uppercase canonical value.
    
    Returns:
        Uppercase single-letter representation (S, Z, Y, etc.)
    """
    return self.value.upper()


# Attach methods to ParameterType class
ParameterType._missing_ = _parameter_type_missing
ParameterType.display_name = property(_parameter_type_display_name)
ParameterType.valid_inputs = _parameter_type_valid_inputs
ParameterType.valid_types = _parameter_type_valid_types
ParameterType.__str__ = _parameter_type_str


# %% [markdown]
# ## PresentationType Enum
#
# Represents how network parameters should be displayed or processed.
# Includes magnitude, phase, complex components, time domain, and special plots.

# %%
class PresentationType(Enum):
    """Network parameter presentation/component types.
    
    Defines how network parameters are presented or visualized. Organized
    into categories: magnitude, phase, complex components, time domain,
    and special plots.
    
    Attributes:
        MAG: Linear magnitude
        DB: Magnitude in decibels (20*log10)
        DB10: Magnitude in decibels (10*log10)
        DEG: Phase in degrees
        DEG_UNWRAP: Unwrapped phase in degrees
        RAD: Phase in radians
        RAD_UNWRAP: Unwrapped phase in radians
        ARCL: Arc length on Smith chart
        ARCL_UNWRAP: Unwrapped arc length
        RE: Real part of complex number
        IM: Imaginary part of complex number
        TIME: Time domain response
        TIME_DB: Time domain in dB
        TIME_MAG: Time domain magnitude
        TIME_IMPULSE: Impulse response
        TIME_STEP: Step response
        VSWR: Voltage Standing Wave Ratio
        SMITH: Smith chart plot
        COMPLEX: Complex plane (Nyquist) plot
        POLAR: Polar plot
    
    Examples:
        >>> PresentationType('mag')
        <PresentationType.MAG: 'mag'>
        >>> PresentationType('DECIBEL')
        <PresentationType.DB: 'db'>
        >>> PresentationType.DB.display_name
        'Magnitude (dB)'
        >>> PresentationType.DEG.category
        'phase'
    
    Notes:
        - Lookups are case-insensitive
        - Use .category property to get grouping
        - Use .display_name for UI-friendly strings
    """
    
    # Tell Python these aren't enum members
    _ignore_ = ['_ALIASES', '_DISPLAY_NAMES', '_CATEGORIES']
    
    # Magnitude and amplitude
    MAG = 'mag'
    DB = 'db'
    DB10 = 'db10'
    
    # Phase/angle
    DEG = 'deg'
    DEG_UNWRAP = 'deg_unwrap'
    RAD = 'rad'
    RAD_UNWRAP = 'rad_unwrap'
    ARCL = 'arcl'
    ARCL_UNWRAP = 'arcl_unwrap'
    
    # Complex components
    RE = 're'
    IM = 'im'
    
    # Time domain
    TIME = 'time'
    TIME_DB = 'time_db'
    TIME_MAG = 'time_mag'
    TIME_IMPULSE = 'time_impulse'
    TIME_STEP = 'time_step'
    
    # Special plots/presentations
    VSWR = 'vswr'
    SMITH = 'smith'
    COMPLEX = 'complex'
    POLAR = 'polar'


# %%
# Define aliases, display names, and categories after the class
PresentationType._ALIASES = {
    # Magnitude
    'mag': 'MAG',
    'magnitude': 'MAG',
    'abs': 'MAG',
    'amplitude': 'MAG',
    'amp': 'MAG',
    
    # dB
    'db': 'DB',
    'decibel': 'DB',
    'decibels': 'DB',
    'log': 'DB',
    'log_mag': 'DB',
    'logmag': 'DB',
    
    # dB10
    'db10': 'DB10',
    'db_10': 'DB10',
    
    # Degrees
    'deg': 'DEG',
    'degree': 'DEG',
    'degrees': 'DEG',
    'phase': 'DEG',
    'angle': 'DEG',
    'phase_deg': 'DEG',
    
    # Degrees unwrapped
    'deg_unwrap': 'DEG_UNWRAP',
    'deg-unwrap': 'DEG_UNWRAP',
    'degunwrap': 'DEG_UNWRAP',
    'degree_unwrap': 'DEG_UNWRAP',
    'degrees_unwrap': 'DEG_UNWRAP',
    'phase_unwrap': 'DEG_UNWRAP',
    'angle_unwrap': 'DEG_UNWRAP',
    'unwrap_deg': 'DEG_UNWRAP',
    'unwrap_degrees': 'DEG_UNWRAP',
    
    # Radians
    'rad': 'RAD',
    'radian': 'RAD',
    'radians': 'RAD',
    'phase_rad': 'RAD',
    
    # Radians unwrapped
    'rad_unwrap': 'RAD_UNWRAP',
    'rad-unwrap': 'RAD_UNWRAP',
    'radunwrap': 'RAD_UNWRAP',
    'radian_unwrap': 'RAD_UNWRAP',
    'radians_unwrap': 'RAD_UNWRAP',
    'unwrap_rad': 'RAD_UNWRAP',
    'unwrap_radians': 'RAD_UNWRAP',
    
    # Arc length
    'arcl': 'ARCL',
    'arc_length': 'ARCL',
    'arclength': 'ARCL',
    'arc-length': 'ARCL',
    
    # Arc length unwrapped
    'arcl_unwrap': 'ARCL_UNWRAP',
    'arcl-unwrap': 'ARCL_UNWRAP',
    'arclunwrap': 'ARCL_UNWRAP',
    'arc_length_unwrap': 'ARCL_UNWRAP',
    'arclength_unwrap': 'ARCL_UNWRAP',
    
    # Real
    're': 'RE',
    'real': 'RE',
    'real_part': 'RE',
    
    # Imaginary
    'im': 'IM',
    'imag': 'IM',
    'imaginary': 'IM',
    'imag_part': 'IM',
    'imaginary_part': 'IM',
    
    # Time domain
    'time': 'TIME',
    'time_domain': 'TIME',
    'timedomain': 'TIME',
    't': 'TIME',
    
    'time_db': 'TIME_DB',
    'time-db': 'TIME_DB',
    'timedb': 'TIME_DB',
    'time_decibel': 'TIME_DB',
    
    'time_mag': 'TIME_MAG',
    'time-mag': 'TIME_MAG',
    'timemag': 'TIME_MAG',
    'time_magnitude': 'TIME_MAG',
    
    'time_impulse': 'TIME_IMPULSE',
    'time-impulse': 'TIME_IMPULSE',
    'timeimpulse': 'TIME_IMPULSE',
    'impulse': 'TIME_IMPULSE',
    'impulse_response': 'TIME_IMPULSE',
    
    'time_step': 'TIME_STEP',
    'time-step': 'TIME_STEP',
    'timestep': 'TIME_STEP',
    'step': 'TIME_STEP',
    'step_response': 'TIME_STEP',
    
    # VSWR
    'vswr': 'VSWR',
    'voltage_standing_wave_ratio': 'VSWR',
    'swr': 'VSWR',
    
    # Smith chart
    'smith': 'SMITH',
    'smith_chart': 'SMITH',
    'smithchart': 'SMITH',
    'smith-chart': 'SMITH',
    
    # Complex plane
    'complex': 'COMPLEX',
    'complex_plane': 'COMPLEX',
    'complexplane': 'COMPLEX',
    'complex-plane': 'COMPLEX',
    'nyquist': 'COMPLEX',
    
    # Polar
    'polar': 'POLAR',
    'polar_plot': 'POLAR',
    'polarplot': 'POLAR',
}

PresentationType._DISPLAY_NAMES = {
    'MAG': 'Magnitude',
    'DB': 'Magnitude (dB)',
    'DB10': 'Magnitude (10*log10)',
    'DEG': 'Phase (degrees)',
    'DEG_UNWRAP': 'Phase Unwrapped (degrees)',
    'RAD': 'Phase (radians)',
    'RAD_UNWRAP': 'Phase Unwrapped (radians)',
    'ARCL': 'Arc Length',
    'ARCL_UNWRAP': 'Arc Length Unwrapped',
    'RE': 'Real Part',
    'IM': 'Imaginary Part',
    'TIME': 'Time Domain',
    'TIME_DB': 'Time Domain (dB)',
    'TIME_MAG': 'Time Domain Magnitude',
    'TIME_IMPULSE': 'Time Domain Impulse Response',
    'TIME_STEP': 'Time Domain Step Response',
    'VSWR': 'Voltage Standing Wave Ratio',
    'SMITH': 'Smith Chart',
    'COMPLEX': 'Complex Plane',
    'POLAR': 'Polar Plot',
}

PresentationType._CATEGORIES = {
    'magnitude': ['MAG', 'DB', 'DB10'],
    'phase': ['DEG', 'DEG_UNWRAP', 'RAD', 'RAD_UNWRAP', 'ARCL', 'ARCL_UNWRAP'],
    'complex': ['RE', 'IM'],
    'time': ['TIME', 'TIME_DB', 'TIME_MAG', 'TIME_IMPULSE', 'TIME_STEP'],
    'special': ['VSWR', 'SMITH', 'COMPLEX', 'POLAR'],
}


# %%
@classmethod
def _presentation_type_missing(cls, value):
    """Handle case-insensitive and alias lookups.
    
    Args:
        value: The value to look up (typically a string)
        
    Returns:
        The matching enum member, or None if not found
    """
    if not isinstance(value, str):
        return None
        
    # Normalize: lowercase and strip whitespace
    normalized = value.lower().strip()
    
    # Try to find in aliases
    canonical_name = cls._ALIASES.get(normalized)
    if canonical_name:
        return cls[canonical_name]
    
    return None


def _presentation_type_display_name(self) -> str:
    """Get user-friendly display name.
    
    Returns:
        Human-readable name suitable for UI display
    """
    return self._DISPLAY_NAMES[self.name]


def _presentation_type_category(self) -> Optional[str]:
    """Get the category this presentation type belongs to.
    
    Returns:
        Category name ('magnitude', 'phase', 'complex', 'time', 'special')
        or None if not categorized
        
    Examples:
        >>> PresentationType.DB.category
        'magnitude'
        >>> PresentationType.DEG.category
        'phase'
    """
    for cat, members in self._CATEGORIES.items():
        if self.name in members:
            return cat
    return None


@classmethod
def _presentation_type_valid_inputs(cls) -> List[str]:
    """Get list of all valid input strings (canonical + aliases).
    
    Returns:
        Sorted list of all accepted input strings
    """
    # Get all aliases
    aliases = set(cls._ALIASES.keys())
    # Add canonical enum values
    canonical = {member.value for member in cls}
    # Add uppercase names
    names = {member.name for member in cls}
    
    return sorted(aliases | canonical | names)


@classmethod
def _presentation_type_valid_types(cls) -> str:
    """Get formatted string of valid presentation types for error messages.
    
    Returns:
        Multi-line string organized by category
    """
    lines = []
    for category, member_names in cls._CATEGORIES.items():
        members = [cls[name].value for name in member_names]
        lines.append(f"  {category.capitalize()}: {', '.join(members)}")
    return '\n'.join(lines)


def _presentation_type_str(self):
    """String representation returns the canonical value.
    
    Returns:
        Lowercase canonical value (mag, db, deg, etc.)
    """
    return self.value


# Attach methods to PresentationType class
PresentationType._missing_ = _presentation_type_missing
PresentationType.display_name = property(_presentation_type_display_name)
PresentationType.category = property(_presentation_type_category)
PresentationType.valid_inputs = _presentation_type_valid_inputs
PresentationType.valid_types = _presentation_type_valid_types
PresentationType.__str__ = _presentation_type_str


# %% [markdown]
# ## MixedModeType Enum
#
# For 4-port differential circuits, represents the mode conversion types.
# Used in high-speed digital design for differential pair analysis.

# %%
class MixedModeType(Enum):
    """Mixed-mode parameter types for 4-port networks.
    
    Represents the four mode conversion combinations in differential/common
    mode analysis. Essential for analyzing differential pairs in high-speed
    digital design (USB, PCIe, HDMI, etc.).
    
    Attributes:
        DD: Differential to Differential (desired signal path)
        DC: Differential to Common (mode conversion, usually unwanted)
        CD: Common to Differential (mode conversion, usually unwanted)
        CC: Common to Common (common mode signal path)
    
    Examples:
        >>> MixedModeType('dd')
        <MixedModeType.DD: 'dd'>
        >>> MixedModeType('DIFFERENTIAL-COMMON')
        <MixedModeType.DC: 'dc'>
        >>> MixedModeType.DC.display_name
        'Differential → Common'
        >>> MixedModeType.DC.is_mode_conversion
        True
    
    Notes:
        - DD and CC are "pure" modes
        - DC and CD indicate mode conversion (typically unwanted)
        - Only applicable to 4-port networks
        - Lookups are case-insensitive
    """
    
    # Tell Python these aren't enum members
    _ignore_ = ['_ALIASES', '_DISPLAY_NAMES']
    
    DD = 'dd'  # Differential to Differential
    DC = 'dc'  # Differential to Common
    CD = 'cd'  # Common to Differential
    CC = 'cc'  # Common to Common


# %%
# Define aliases after the class
MixedModeType._ALIASES = {
    # DD - Differential to Differential
    'dd': 'DD',
    'd-d': 'DD',
    'd_d': 'DD',
    'diff-diff': 'DD',
    'diff_diff': 'DD',
    'diffdiff': 'DD',
    'differential-differential': 'DD',
    'differential_differential': 'DD',
    'differentialdifferential': 'DD',
    'diff-to-diff': 'DD',
    'diff_to_diff': 'DD',
    'difftodiff': 'DD',
    
    # DC - Differential to Common
    'dc': 'DC',
    'd-c': 'DC',
    'd_c': 'DC',
    'diff-comm': 'DC',
    'diff_comm': 'DC',
    'diffcomm': 'DC',
    'diff-common': 'DC',
    'diff_common': 'DC',
    'diffcommon': 'DC',
    'differential-common': 'DC',
    'differential_common': 'DC',
    'differentialcommon': 'DC',
    'diff-to-comm': 'DC',
    'diff_to_comm': 'DC',
    'difftocomm': 'DC',
    'diff-to-common': 'DC',
    'diff_to_common': 'DC',
    'difftocommon': 'DC',
    
    # CD - Common to Differential
    'cd': 'CD',
    'c-d': 'CD',
    'c_d': 'CD',
    'comm-diff': 'CD',
    'comm_diff': 'CD',
    'commdiff': 'CD',
    'common-diff': 'CD',
    'common_diff': 'CD',
    'commondiff': 'CD',
    'common-differential': 'CD',
    'common_differential': 'CD',
    'commondifferential': 'CD',
    'comm-to-diff': 'CD',
    'comm_to_diff': 'CD',
    'commtodiff': 'CD',
    'common-to-diff': 'CD',
    'common_to_diff': 'CD',
    'commontodiff': 'CD',
    'common-to-differential': 'CD',
    'common_to_differential': 'CD',
    'commontodifferential': 'CD',
    
    # CC - Common to Common
    'cc': 'CC',
    'c-c': 'CC',
    'c_c': 'CC',
    'comm-comm': 'CC',
    'comm_comm': 'CC',
    'commcomm': 'CC',
    'common-common': 'CC',
    'common_common': 'CC',
    'commoncommon': 'CC',
    'comm-to-comm': 'CC',
    'comm_to_comm': 'CC',
    'commtocomm': 'CC',
}

MixedModeType._DISPLAY_NAMES = {
    'DD': 'Differential → Differential',
    'DC': 'Differential → Common',
    'CD': 'Common → Differential',
    'CC': 'Common → Common',
}


# %%
@classmethod
def _mixed_mode_type_missing(cls, value):
    """Handle case-insensitive and alias lookups.
    
    Args:
        value: The value to look up (typically a string)
        
    Returns:
        The matching enum member, or None if not found
    """
    if not isinstance(value, str):
        return None
        
    # Normalize: lowercase and strip whitespace
    normalized = value.lower().strip()
    
    # Try to find in aliases
    canonical_name = cls._ALIASES.get(normalized)
    if canonical_name:
        return cls[canonical_name]
    
    return None


def _mixed_mode_type_display_name(self) -> str:
    """Get user-friendly display name with arrow notation.
    
    Returns:
        Human-readable name like 'Differential → Common'
    """
    return self._DISPLAY_NAMES[self.name]


def _mixed_mode_type_is_mode_conversion(self) -> bool:
    """Check if this represents mode conversion (DC or CD).
    
    Mode conversion parameters indicate unwanted coupling between
    differential and common modes. High mode conversion typically
    indicates design problems.
    
    Returns:
        True if mode conversion (DC or CD), False for pure modes (DD or CC)
        
    Examples:
        >>> MixedModeType.DC.is_mode_conversion
        True
        >>> MixedModeType.DD.is_mode_conversion
        False
    """
    return self in (MixedModeType.DC, MixedModeType.CD)


@classmethod
def _mixed_mode_type_valid_inputs(cls) -> List[str]:
    """Get list of all valid input strings (canonical + aliases).
    
    Returns:
        Sorted list of all accepted input strings
    """
    # Get all aliases
    aliases = set(cls._ALIASES.keys())
    # Add canonical enum values
    canonical = {member.value for member in cls}
    # Add uppercase names
    names = {member.name for member in cls}
    
    return sorted(aliases | canonical | names)


@classmethod
def _mixed_mode_type_valid_types(cls) -> str:
    """Get formatted string of valid mixed-mode types for error messages.
    
    Returns:
        Comma-separated string like "DD, DC, CD, CC"
    """
    return ', '.join([member.name for member in cls])


def _mixed_mode_type_str(self):
    """String representation returns uppercase canonical value.
    
    Returns:
        Uppercase representation (DD, DC, CD, CC)
    """
    return self.value.upper()


# Attach methods to MixedModeType class
MixedModeType._missing_ = _mixed_mode_type_missing
MixedModeType.display_name = property(_mixed_mode_type_display_name)
MixedModeType.is_mode_conversion = property(_mixed_mode_type_is_mode_conversion)
MixedModeType.valid_inputs = _mixed_mode_type_valid_inputs
MixedModeType.valid_types = _mixed_mode_type_valid_types
MixedModeType.__str__ = _mixed_mode_type_str


# %% [markdown]
# ## Custom Exception Classes
#
# These exceptions provide helpful error messages when invalid enum values
# are provided, automatically showing all valid options and common aliases.

# %%
class ParameterTypeError(ValueError):
    """Raised when an invalid parameter type is provided.
    
    Automatically formats error message with valid options from the
    ParameterType enum, including all accepted aliases.
    
    Args:
        invalid_input: The invalid value that was provided
        argument_name: Name of the argument for error message context
    
    Examples:
        >>> try:
        ...     param = ParameterType('invalid')
        ... except (ValueError, KeyError):
        ...     raise ParameterTypeError('invalid', 'representation')
    """
    
    def __init__(self, invalid_input, argument_name: str = "representation"):
        """Initialize with helpful error message.
        
        Args:
            invalid_input: The value that failed validation
            argument_name: Name of the parameter (for context in error message)
        """
        self.invalid_input = invalid_input
        self.argument_name = argument_name
        
        valid_types = ParameterType.valid_types()
        
        message = (
            f"`{argument_name}` invalid input: {invalid_input!r}\n"
            f"Valid parameter types: {valid_types}\n"
            f"\nAccepted aliases (case-insensitive):\n"
            f"  S: s, scattering, s-parameters, s-params\n"
            f"  Z: z, impedance, z-parameters\n"
            f"  Y: y, admittance, y-parameters\n"
            f"  A: a, abcd, transmission\n"
            f"  T: t, transfer, scattering-transfer\n"
            f"  H: h, hybrid, h-parameters\n"
            f"  G: g, inverse-hybrid, g-parameters"
        )
        
        super().__init__(message)


# %%
class PresentationTypeError(ValueError):
    """Raised when an invalid presentation type is provided.
    
    Automatically formats error message with valid options from the
    PresentationType enum, organized by category.
    
    Args:
        invalid_input: The invalid value that was provided
        argument_name: Name of the argument for error message context
    
    Examples:
        >>> try:
        ...     pres = PresentationType('invalid')
        ... except (ValueError, KeyError):
        ...     raise PresentationTypeError('invalid', 'presentation')
    """
    
    def __init__(self, invalid_input, argument_name: str = "presentation"):
        """Initialize with helpful error message.
        
        Args:
            invalid_input: The value that failed validation
            argument_name: Name of the parameter (for context in error message)
        """
        self.invalid_input = invalid_input
        self.argument_name = argument_name
        
        valid_types = PresentationType.valid_types()
        
        message = (
            f"`{argument_name}` invalid input: {invalid_input!r}\n"
            f"Valid presentation types:\n"
            f"{valid_types}\n"
            f"\nCommon aliases (case-insensitive):\n"
            f"  Magnitude: mag, magnitude, abs, amplitude\n"
            f"  Phase: deg, degrees, phase, angle, rad, radians\n"
            f"  Complex: re, real, im, imag, imaginary\n"
            f"  Special: smith, vswr, polar, complex"
        )
        
        super().__init__(message)


# %%
class MixedModeTypeError(ValueError):
    """Raised when an invalid mixed-mode type is provided.
    
    Automatically formats error message with valid options from the
    MixedModeType enum. Reminds user that mixed-mode is for 4-port networks.
    
    Args:
        invalid_input: The invalid value that was provided
        argument_name: Name of the argument for error message context
    
    Examples:
        >>> try:
        ...     mode = MixedModeType('invalid')
        ... except (ValueError, KeyError):
        ...     raise MixedModeTypeError('invalid', 'mode')
    """
    
    def __init__(self, invalid_input, argument_name: str = "mode"):
        """Initialize with helpful error message.
        
        Args:
            invalid_input: The value that failed validation
            argument_name: Name of the parameter (for context in error message)
        """
        self.invalid_input = invalid_input
        self.argument_name = argument_name
        
        valid_types = MixedModeType.valid_types()
        
        message = (
            f"`{argument_name}` invalid input: {invalid_input!r}\n"
            f"Valid mixed-mode types (4-port networks): {valid_types}\n"
            f"\nAccepted aliases (case-insensitive):\n"
            f"  DD: dd, diff-diff, differential-differential\n"
            f"  DC: dc, diff-comm, differential-common (mode conversion)\n"
            f"  CD: cd, comm-diff, common-differential (mode conversion)\n"
            f"  CC: cc, comm-comm, common-common"
        )
        
        super().__init__(message)


# %% [markdown]
# ## Usage Examples and Tests
#
# Demonstrates the key features of each enum type with practical examples.

# %%
def main():
    """Run demonstration of enum functionality."""
    
    print("=" * 70)
    print("TOUCHSTONE ANALYZER - ENUMS DEMONSTRATION")
    print("=" * 70)
    
    # ========================================================================
    # ParameterType Examples
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. PARAMETERTYPE - Network Parameter Representations")
    print("=" * 70)
    
    print("\n--- Basic Usage ---")
    print(f"ParameterType('s') = {ParameterType('s')}")
    print(f"ParameterType('SCATTERING') = {ParameterType('SCATTERING')}")
    print(f"ParameterType('z-parameters') = {ParameterType('z-parameters')}")
    
    print("\n--- Display Names ---")
    print(f"ParameterType.S.display_name = '{ParameterType.S.display_name}'")
    print(f"ParameterType.Z.display_name = '{ParameterType.Z.display_name}'")
    
    print("\n--- String Conversion ---")
    print(f"str(ParameterType.S) = '{str(ParameterType.S)}'")
    print(f"str(ParameterType.Z) = '{str(ParameterType.Z)}'")
    
    print("\n--- All Valid Types ---")
    print(f"Valid types: {ParameterType.valid_types()}")
    
    # ========================================================================
    # PresentationType Examples
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. PRESENTATIONTYPE - Data Presentation Formats")
    print("=" * 70)
    
    print("\n--- Basic Usage ---")
    print(f"PresentationType('mag') = {PresentationType('mag')}")
    print(f"PresentationType('DECIBEL') = {PresentationType('DECIBEL')}")
    print(f"PresentationType('phase') = {PresentationType('phase')}")
    
    print("\n--- Display Names and Categories ---")
    print(f"PresentationType.DB.display_name = '{PresentationType.DB.display_name}'")
    print(f"PresentationType.DB.category = '{PresentationType.DB.category}'")
    print(f"PresentationType.DEG.display_name = '{PresentationType.DEG.display_name}'")
    print(f"PresentationType.DEG.category = '{PresentationType.DEG.category}'")
    
    print("\n--- Category Breakdown ---")
    for category, members in PresentationType._CATEGORIES.items():
        member_names = [PresentationType[m].value for m in members[:3]]  # Show first 3
        more = f" (+{len(members)-3} more)" if len(members) > 3 else ""
        print(f"  {category.capitalize()}: {', '.join(member_names)}{more}")
    
    # ========================================================================
    # MixedModeType Examples
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. MIXEDMODETYPE - 4-Port Differential Analysis")
    print("=" * 70)
    
    print("\n--- Basic Usage ---")
    print(f"MixedModeType('dd') = {MixedModeType('dd')}")
    print(f"MixedModeType('DIFF-COMM') = {MixedModeType('DIFF-COMM')}")
    print(f"MixedModeType('common-differential') = {MixedModeType('common-differential')}")
    
    print("\n--- Display Names and Mode Conversion ---")
    for mode in MixedModeType:
        conversion = " ⚠️ MODE CONVERSION" if mode.is_mode_conversion else ""
        print(f"{str(mode):3} - {mode.display_name:30} {conversion}")
    
    # ========================================================================
    # Validation Pattern Example
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. VALIDATION PATTERN - How to Use in Your Code")
    print("=" * 70)
    
    def plot_trace(representation='S', presentation='mag'):
        """Example function showing validation pattern.
        
        Args:
            representation: Parameter type (S, Z, Y, etc.)
            presentation: Presentation format (mag, db, phase, etc.)
        
        Returns:
            Tuple of validated enum values
        
        Raises:
            ParameterTypeError: If representation is invalid
            PresentationTypeError: If presentation is invalid
        """
        # Validate inputs with helpful error messages
        try:
            param_type = ParameterType(representation)
        except (ValueError, KeyError):
            raise ParameterTypeError(representation, "representation")
        
        try:
            pres_type = PresentationType(presentation)
        except (ValueError, KeyError):
            raise PresentationTypeError(presentation, "presentation")
        
        print(f"✓ Plotting {param_type.display_name} as {pres_type.display_name}")
        return param_type, pres_type
    
    # Test valid inputs
    print("\n--- Valid Inputs ---")
    print("plot_trace('scattering', 'db'):")
    plot_trace('scattering', 'db')
    
    print("\nplot_trace('z', 'phase'):")
    plot_trace('z', 'phase')
    
    # Test invalid input
    print("\n--- Invalid Input Handling ---")
    print("plot_trace('invalid', 'mag'):")
    try:
        plot_trace('invalid', 'mag')
    except ParameterTypeError as e:
        print(f"❌ Caught ParameterTypeError:")
        print(f"   {str(e).split(chr(10))[0]}")  # Just first line
        print("   (Full error message provides all valid options and aliases)")
    
    # ========================================================================
    # Mixed-Mode Validation Example
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. MIXED-MODE VALIDATION EXAMPLE")
    print("=" * 70)
    
    def plot_mixed_mode(mode='dd', representation='S', presentation='mag'):
        """Example for 4-port mixed-mode plotting.
        
        Args:
            mode: Mixed-mode type (DD, DC, CD, CC)
            representation: Parameter type (S, Z, Y, etc.)
            presentation: Presentation format (mag, db, phase, etc.)
        
        Returns:
            Tuple of validated enum values
        """
        try:
            mode_type = MixedModeType(mode)
        except (ValueError, KeyError):
            raise MixedModeTypeError(mode, "mode")
        
        try:
            param_type = ParameterType(representation)
        except (ValueError, KeyError):
            raise ParameterTypeError(representation, "representation")
        
        try:
            pres_type = PresentationType(presentation)
        except (ValueError, KeyError):
            raise PresentationTypeError(presentation, "presentation")
        
        conversion_note = " ⚠️ MODE CONVERSION" if mode_type.is_mode_conversion else ""
        print(f"✓ Plotting {mode_type.display_name}{conversion_note}")
        print(f"  {param_type.display_name} as {pres_type.display_name}")
        return mode_type, param_type, pres_type
    
    print("\n--- Valid 4-Port Examples ---")
    print("plot_mixed_mode('diff-comm', 's', 'db'):")
    plot_mixed_mode('diff-comm', 's', 'db')
    
    print("\nplot_mixed_mode('CC', 'z', 'phase'):")
    plot_mixed_mode('CC', 'z', 'phase')
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Features:
  • Case-insensitive lookups work seamlessly
  • Extensive aliases for user convenience
  • Helpful error messages with all valid options
  • Display names for UI/reporting
  • Category grouping for PresentationType
  • Mode conversion detection for MixedModeType

Integration Tips:
  • Always validate user input with try/except
  • Use custom exceptions for helpful error messages
  • Use .display_name for UI elements
  • Use str(enum) for short canonical representation
  • Check .is_mode_conversion for MixedModeType analysis
    """)


# %% [markdown]
# ## Run Main Demonstration
#
# Execute the demonstration when run as a script or notebook.

# %%
if __name__ == "__main__":
    main()
