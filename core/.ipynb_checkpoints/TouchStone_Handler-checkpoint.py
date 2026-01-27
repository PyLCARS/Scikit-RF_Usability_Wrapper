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
# # Touchstone Data Handler
#
# Comprehensive handler for Touchstone files (VNA S-parameter data) with advanced features.
#
# ## Key Features
# - **Metadata Management**: VNA settings, calibration info, measurement conditions
# - **DC Extrapolation**: Improve time domain analysis with DC point extrapolation
# - **Time Domain Calculations**: Automatic resolution and range computation
# - **File I/O**: Read/write with metadata preservation
# - **attrs Integration**: Clean data class with validation
#
# ## Quick Example
# ```python
# # Load touchstone file
# data = TouchstoneData(
#     file_folder=Path('.'),
#     file_name='my_measurement',
#     num_ports=2
# )
#
# # Access data
# print(data.frequency_scaled)  # Frequency in GHz/MHz (auto-scaled)
# print(data.network_single_ended.s)  # S-parameters
#
# # Modify metadata
# data.instrument_manufacturer = "Keysight"
# data.velocity_factor = 0.7  # For coax
#
# # Write with metadata
# data.write_touchstone(Path('output.s2p'))
# ```
#
# Initial Author: GProtoZeroW - Jan 2026

# %% [markdown]
# ## Imports

# %%
import skrf as rf
import numpy as np
from pathlib import Path
import attrs
from typing import Optional
from scipy import constants
import sys

# %% [markdown]
# ## Logging Setup
#
# Flexible logging that works standalone or as part of larger package.
# Falls back to standard logging if loguru not available.

# %%
try:
    from loguru import logger
    # Check if logger is already configured (e.g., by parent package)
    if not logger._core.handlers:
        # Not configured - set up for standalone use
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        logger.info("Loguru configured for standalone use")
except ImportError:
    # Fallback to standard logging if loguru not available
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    logger.info("Using standard logging (loguru not available)")

# %% [markdown]
# ## NumPy Warning Suppression
#
# Suppress divide-by-zero warnings that can occur in RF calculations.

# %%
np.seterr(divide='ignore', invalid='ignore')

# %% [markdown]
# ## TouchstoneData Class
#
# Main data class using attrs for clean, validated structure.
#
# ### Architecture
# - **attrs-based**: Automatic `__init__`, `__repr__`, validation
# - **Frozen fields**: Some fields are read-only after initialization (`on_setattr=attrs.setters.NO_OP`)
# - **Metadata**: Each field has 'help' text for tooltips/documentation
# - **Lazy computation**: Frequency scaling and time domain params computed from network data
#
# ### Field Categories
# 1. **Initialization**: file_folder, file_name, num_ports
# 2. **Instrument ID**: IDN, manufacturer, model, serial, firmware
# 3. **Port Config**: port_name_map
# 4. **Frequency**: Auto-computed from network (start, stop, center, span, npoints)
# 5. **VNA Settings**: IF bandwidth, averaging, power, calibration
# 6. **Measurement**: Date, time
# 7. **Velocity Factor**: For time domain
# 8. **File Metadata**: Version, format, comments
# 9. **Network Data**: scikit-rf Network objects
# 10. **DC Extrapolation**: Settings and extrapolated network
# 11. **Time Domain**: Resolution and range (auto-computed)

# %%
@attrs.define
class TouchstoneData:
    """Handle touchstone S-parameter files with metadata and DC extrapolation.
    
    Provides comprehensive management of Touchstone files including VNA metadata,
    DC extrapolation for time domain analysis, and convenient file I/O.
    
    Parameters
    ----------
    file_folder : Path
        Directory containing the touchstone file
    file_name : Path
        Filename without extension (e.g., "my_measurement")
    num_ports : int
        Number of ports (determines .s1p, .s2p, etc.)
    dc_extrap_points : int, optional
        Number of DC extrapolation points (default: None = auto)
    dc_extrap_dc_sparam : ndarray, optional
        Target S-param values at DC (default: None = extrapolate)
    dc_extrap_kind : str, optional
        Interpolation: "linear", "cubic", "quadratic" (default: None = cubic)
    dc_extrap_coords : str, optional
        Coordinates: "cart" or "polar" (default: None = cart)
    
    Attributes
    ----------
    network_single_ended : rf.Network
        Original S-parameter data from file
    network_dc_extrapolated : rf.Network
        DC-extrapolated network for time domain analysis
    frequency : ndarray
        Frequency array in Hz
    frequency_scaled : ndarray
        Frequency in auto-selected units (MHz, GHz, etc.)
    time_resolution_s : float
        Time domain resolution in seconds
    max_time_range_s : float
        Maximum time domain range in seconds (one-way)
    
    Examples
    --------
    Load file:
    
    >>> data = TouchstoneData(
    ...     file_folder=Path('.'),
    ...     file_name='my_measurement',
    ...     num_ports=2
    ... )
    
    Access data:
    
    >>> data.frequency_scaled  # Auto-scaled frequency array
    >>> data.network_single_ended.s  # S-parameters
    >>> data.time_resolution_s * 1e12  # Resolution in ps
    
    Modify and save:
    
    >>> data.instrument_manufacturer = "Keysight"
    >>> data.velocity_factor = 0.7
    >>> data.write_touchstone(Path('output.s2p'))
    """
    
    # ==========================================
    # INITIALIZATION PARAMETERS
    # ==========================================
    file_folder: Path = attrs.field(
        converter=Path,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Directory containing the touchstone file'}
    )
    file_name: Path = attrs.field(
        converter=Path,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Filename without extension'}
    )
    num_ports: int = attrs.field(
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Number of ports (1-N)'}
    )
    
    # ==========================================
    # VNA INSTRUMENT IDENTIFICATION
    # ==========================================
    instrument_idn: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Full *IDN? query response'}
    )
    instrument_manufacturer: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Manufacturer (e.g., "Keysight")'}
    )
    instrument_model: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Model number'}
    )
    instrument_serial: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Serial number'}
    )
    instrument_firmware: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Firmware version'}
    )
    
    # ==========================================
    # PORT CONFIGURATION
    # ==========================================
    port_name_map: dict = attrs.field(
        init=False,
        factory=dict,
        repr=lambda d: f"{{{', '.join(f'{k}: {v}' for k, v in list(d.items())[:3])}{', ...' if len(d) > 3 else ''}}}" if d else "{}",
        metadata={'help': 'Port index to name mapping {0: "Input", 1: "Output"}'}
    )
    
    # ==========================================
    # FREQUENCY SETTINGS (Auto-computed)
    # ==========================================
    frequency_unit: str = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Auto-selected unit (hz, khz, mhz, ghz)'}
    )
    start_freq: float = attrs.field(
        init=False,
        repr=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Start frequency (Hz)'}
    )
    start_freq_scaled: float = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Start frequency (scaled unit)'}
    )
    stop_freq: float = attrs.field(
        init=False,
        repr=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Stop frequency (Hz)'}
    )
    stop_freq_scaled: float = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Stop frequency (scaled unit)'}
    )
    center_freq: float = attrs.field(
        init=False,
        repr=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Center frequency (Hz)'}
    )
    center_freq_scaled: float = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Center frequency (scaled unit)'}
    )
    span_freq: float = attrs.field(
        init=False,
        repr=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Frequency span (Hz)'}
    )
    span_freq_scaled: float = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Frequency span (scaled unit)'}
    )
    npoints: int = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Number of frequency points'}
    )
    freq_step: float = attrs.field(
        init=False,
        repr=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Frequency step (Hz)'}
    )
    freq_step_scaled: float = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Frequency step (scaled unit)'}
    )
    sweep_type: str = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Sweep type (lin, log)'}
    )
    
    # ==========================================
    # VNA CONFIGURATION
    # ==========================================
    if_bandwidth_hz: float = attrs.field(
        init=False,
        default=1000.0,
        metadata={'vna_meta': True, 'help': 'IF bandwidth (Hz) - lower = less noise, slower sweep'}
    )
    num_averages: int = attrs.field(
        init=False,
        default=1,
        metadata={'vna_meta': True, 'help': 'Number of averages - higher = less noise'}
    )
    averaging_enabled: bool = attrs.field(
        init=False,
        default=False,
        metadata={'vna_meta': True, 'help': 'Was averaging enabled?'}
    )
    source_power_dbm: float = attrs.field(
        init=False,
        default=0.0,
        metadata={'vna_meta': True, 'help': 'Source power (dBm)'}
    )
    port_powers: dict = attrs.field(
        init=False,
        factory=dict,
        metadata={'vna_meta': True, 'help': 'Per-port powers {port: power_dbm}'}
    )
    
    # ==========================================
    # CALIBRATION
    # ==========================================
    calibration_type: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Cal method (SOLT, TRL, etc.)'}
    )
    calibration_date: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Calibration date (ISO format)'}
    )
    calibration_kit: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Cal kit used'}
    )
    reference_impedance: float = attrs.field(
        init=False,
        default=50.0,
        metadata={'vna_meta': True, 'help': 'Reference Z₀ (Ω)'}
    )
    
    # ==========================================
    # TRIGGER & SWEEP
    # ==========================================
    trigger_source: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Trigger source'}
    )
    sweep_time_s: float = attrs.field(
        init=False,
        default=0.0,
        metadata={'vna_meta': True, 'help': 'Sweep time (seconds)'}
    )
    
    # ==========================================
    # MEASUREMENT DATE/TIME
    # ==========================================
    measurement_date: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Measurement date (YYYY-MM-DD)'}
    )
    measurement_time: str = attrs.field(
        init=False,
        default="",
        metadata={'vna_meta': True, 'help': 'Measurement time (HH:MM:SS)'}
    )
    
    # ==========================================
    # VELOCITY FACTOR
    # ==========================================
    velocity_factor: float = attrs.field(
        init=False,
        default=1.0,
        metadata={'vna_meta': True, 'help': 'Vp for time domain (1.0=free space, ~0.66-0.7=coax, ~0.85=microstrip)'}
    )
    
    # ==========================================
    # FILE METADATA
    # ==========================================
    full_file_path: Path = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Complete file path with .sNp extension'}
    )
    touchstone_version: str = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Touchstone format version (1.0, 1.1, 2.0)'}
    )
    touchstone_format: str = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Data format (RI, MA, DB)'}
    )
    comments: str = attrs.field(
        init=False,
        factory=str,
        repr=lambda c: f"'{c[:50]}...' ({len(c)} chars)" if len(c) > 50 else f"'{c}'",
        metadata={'help': 'Comments from file (! lines)'}
    )
    
    # ==========================================
    # NETWORK DATA
    # ==========================================
    network_single_ended: rf.Network = attrs.field(
        init=False,
        repr=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Original S-parameter network'}
    )
    
    # ==========================================
    # DC EXTRAPOLATION SETTINGS
    # ==========================================
    dc_extrap_points: Optional[int] = attrs.field(
        default=None,
        repr=lambda x: x if x is not None else attrs.NOTHING,
        metadata={'help': 'Number of DC points to add (None=auto)'}
    )
    dc_extrap_dc_sparam: Optional[np.ndarray] = attrs.field(
        default=None,
        repr=False,
        metadata={'help': 'Target S-params at DC (None=extrapolate)'}
    )
    dc_extrap_kind: Optional[str] = attrs.field(
        default=None,
        repr=lambda x: x if x is not None else attrs.NOTHING,
        metadata={'help': 'Interpolation: linear, cubic, quadratic (None=cubic)'}
    )
    dc_extrap_coords: Optional[str] = attrs.field(
        default=None,
        repr=lambda x: x if x is not None else attrs.NOTHING,
        metadata={'help': 'Coords: cart, polar (None=cart)'}
    )
    
    # ==========================================
    # DC EXTRAPOLATED NETWORK
    # ==========================================
    network_dc_extrapolated: Optional[rf.Network] = attrs.field(
        init=False,
        default=None,
        repr=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'DC-extrapolated network for TDR/TDT'}
    )
    
    # ==========================================
    # TIME DOMAIN PARAMETERS
    # ==========================================
    time_resolution_s: float = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Time resolution (s) = Vp*c / (2 * (Fmax - Fmin))'}
    )
    max_time_range_s: float = attrs.field(
        init=False,
        on_setattr=attrs.setters.NO_OP,
        metadata={'help': 'Max time range (s, one-way) = Vp*c / (2 * Δf)'}
    )
    
    def __attrs_post_init__(self):
        """Post-initialization: Load file if called directly."""
        logger.debug(f"Initializing TouchstoneData: {self.file_name} ({self.num_ports} ports)")
        
        # Initialize default port names
        if not self.port_name_map:
            default_map = {i: f"Port_{i+1}" for i in range(self.num_ports)}
            object.__setattr__(self, 'port_name_map', default_map)
        
        # If network not set, we're being called directly (not via classmethod)
        if not hasattr(self, 'network_single_ended') or self.network_single_ended is None:
            logger.info("Loading via direct instantiation")
            self._load_from_touchstone_()
    
    def _load_from_touchstone_(self):
        """Internal: Load file and compute all derived fields."""
        logger.debug("Starting file load sequence")
        
        # Build full path
        self._make_full_file_path_()
        
        # Load via scikit-rf
        logger.info(f"Loading: {self.full_file_path}")
        try:
            network = rf.Network(str(self.full_file_path))
            object.__setattr__(self, 'network_single_ended', network)
            logger.success(f"Loaded {network.nports} ports, {len(network.f)} points")
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise
        
        # Verify port count matches
        self._verify_port_count_()
        
        # Extract comments and metadata
        self._populate_comments_()
        self._extract_metadata_from_comments()
        
        # Compute frequency parameters
        self._compute_from_network()
        
        # DC extrapolation and time domain
        self._finalize_setup()
        
        logger.success("Initialization complete")
    
    # ==========================================
    # INTERNAL HELPER METHODS
    # ==========================================
    
    def _make_full_file_path_(self):
        """Build full file path with .sNp extension."""
        extension = f".s{self.num_ports}p"
        full_path = self.file_folder / (str(self.file_name) + extension)
        object.__setattr__(self, 'full_file_path', full_path)
        logger.debug(f"Full path: {full_path}")
    
    def _verify_port_count_(self):
        """Verify loaded network matches expected port count."""
        if self.network_single_ended.nports != self.num_ports:
            raise ValueError(
                f"Port mismatch: expected {self.num_ports}, "
                f"got {self.network_single_ended.nports}"
            )
    
    def _populate_comments_(self):
        """Extract comments from touchstone file."""
        try:
            comments = self.network_single_ended.comments
            object.__setattr__(self, 'comments', comments if comments else "")
            logger.debug(f"Extracted {len(self.comments)} chars of comments")
        except Exception as e:
            logger.warning(f"Could not extract comments: {e}")
            object.__setattr__(self, 'comments', "")
    
    def _extract_metadata_from_comments(self):
        """Parse VNA metadata from comment lines."""
        # This would parse custom comment formats - implement based on your needs
        # Example: look for "! VNA_MANUFACTURER: Keysight"
        logger.debug("Extracting metadata from comments")
        # TODO: Implement metadata parsing
        pass
    
    def _compute_from_network(self):
        """Compute all frequency-related parameters from network."""
        logger.debug("Computing frequency parameters")
        
        network = self.network_single_ended
        
        # Touchstone format info
        object.__setattr__(self, 'touchstone_version', '1.0')  # scikit-rf doesn't expose this
        object.__setattr__(self, 'touchstone_format', 'RI')  # Default assumption
        
        # Frequency parameters
        freq = network.frequency
        object.__setattr__(self, 'frequency_unit', freq.unit)
        object.__setattr__(self, 'start_freq', freq.start)
        object.__setattr__(self, 'start_freq_scaled', freq.start_scaled)
        object.__setattr__(self, 'stop_freq', freq.stop)
        object.__setattr__(self, 'stop_freq_scaled', freq.stop_scaled)
        object.__setattr__(self, 'center_freq', freq.center)
        object.__setattr__(self, 'center_freq_scaled', freq.center_scaled)
        object.__setattr__(self, 'span_freq', freq.span)
        object.__setattr__(self, 'span_freq_scaled', freq.span_scaled)
        object.__setattr__(self, 'npoints', freq.npoints)
        object.__setattr__(self, 'freq_step', freq.step)
        object.__setattr__(self, 'freq_step_scaled', freq.step_scaled)
        object.__setattr__(self, 'sweep_type', freq.sweep_type)
        
        logger.debug(f"Frequency: {freq.start_scaled:.3f} - {freq.stop_scaled:.3f} {freq.unit} ({freq.npoints} pts)")
    
    def _finalize_setup(self):
        """Generate DC extrapolation and compute time domain parameters."""
        logger.debug("Finalizing setup")
        
        # Generate DC extrapolation
        self.generate_dc_extrapolation()
        
        # Compute time domain parameters from DC-extrapolated network
        network = self.network_dc_extrapolated if self.network_dc_extrapolated else self.network_single_ended
        self._calc_time_domain_params(network)
    
    def _calc_time_domain_params(self, network):
        """Calculate time domain resolution and range.
        
        Based on:
        - "Correlation between VNA and TDR/TDT Extracted S-Parameters up to 20 GHz"
          (Cherry Wakayama, Jeff Loyer, Intel/UW)
        - "TIME DOMAIN ANALYSIS WITH COPPER MOUNTAIN TECHNOLOGIES VNA"
          (Signal Integrity Tips, Dec 2019)
        """
        logger.debug("Calculating time domain parameters")
        
        # Speed of light with velocity factor
        c = constants.c  # m/s
        v = c * self.velocity_factor
        
        # Frequency parameters
        f_max = network.f.max()
        f_min = network.f.min()
        n_points = len(network.f)
        delta_f = (f_max - f_min) / (n_points - 1) if n_points > 1 else f_max - f_min
        
        # Time domain resolution: v / (2 * (Fmax - Fmin))
        time_res = v / (2 * (f_max - f_min))
        object.__setattr__(self, 'time_resolution_s', time_res)
        
        # Max time range (one way): v / (2 * Δf)
        max_time = v / (2 * delta_f) if delta_f > 0 else 0
        object.__setattr__(self, 'max_time_range_s', max_time)
        
        logger.debug(f"Time params: res={time_res*1e12:.3f} ps, range={max_time*1e9:.3f} ns, Vp={self.velocity_factor}")
    
    # ==========================================
    # DC EXTRAPOLATION
    # ==========================================
    
    def generate_dc_extrapolation(self, points=None, kind=None, coords=None, dc_sparam=None):
        """Generate DC-extrapolated network for improved time domain analysis.
        
        DC extrapolation adds points down to 0 Hz to reduce time domain artifacts
        and improve TDR/TDT analysis accuracy.
        
        Parameters
        ----------
        points : int, optional
            Number of DC points to add (None = use default or stored value)
        kind : str, optional
            Interpolation method: 'linear', 'cubic', 'quadratic' (None = cubic)
        coords : str, optional
            Coordinate system: 'cart' or 'polar' (None = cart)
        dc_sparam : ndarray, optional
            Target S-parameter values at DC (None = extrapolate from data)
        
        Examples
        --------
        >>> data.generate_dc_extrapolation(points=50, kind='linear')
        >>> data.generate_dc_extrapolation(coords='polar')  # Regenerate with polar
        """
        logger.info("Generating DC extrapolation")
        
        # Use provided values or fall back to stored/defaults
        if points is not None:
            object.__setattr__(self, 'dc_extrap_points', points)
        if kind is not None:
            object.__setattr__(self, 'dc_extrap_kind', kind)
        if coords is not None:
            object.__setattr__(self, 'dc_extrap_coords', coords)
        if dc_sparam is not None:
            object.__setattr__(self, 'dc_extrap_dc_sparam', dc_sparam)
        
        # Build kwargs for scikit-rf
        kwargs = {}
        if self.dc_extrap_points is not None:
            kwargs['dc_sparam_n_terms'] = self.dc_extrap_points
        if self.dc_extrap_dc_sparam is not None:
            kwargs['dc_sparam'] = self.dc_extrap_dc_sparam
        if self.dc_extrap_kind is not None:
            kwargs['kind'] = self.dc_extrap_kind
        if self.dc_extrap_coords is not None:
            kwargs['coords'] = self.dc_extrap_coords
        
        logger.debug(f"DC extrap kwargs: {kwargs}")
        
        try:
            network_dc = self.network_single_ended.extrapolate_to_dc(**kwargs)
            object.__setattr__(self, 'network_dc_extrapolated', network_dc)
            
            orig_pts = len(self.network_single_ended.f)
            new_pts = len(network_dc.f)
            added_pts = new_pts - orig_pts
            
            logger.success(f"DC extrapolation complete: added {added_pts} points (total {new_pts})")
        except Exception as e:
            logger.error(f"DC extrapolation failed: {e}")
            object.__setattr__(self, 'network_dc_extrapolated', None)
    
    # ==========================================
    # PROPERTIES
    # ==========================================
    
    @property
    def frequency(self):
        """Frequency array (Hz) from original network."""
        return self.network_single_ended.f
    
    @property
    def frequency_scaled(self):
        """Frequency array in auto-selected units (MHz, GHz, etc.)."""
        return self.network_single_ended.frequency.f_scaled
    
    @property
    def frequency_dc_extrap(self):
        """Frequency array (Hz) from DC-extrapolated network."""
        if self.network_dc_extrapolated is not None:
            return self.network_dc_extrapolated.f
        return None
    
    @property
    def frequency_dc_extrap_scaled(self):
        """Frequency array (scaled) from DC-extrapolated network."""
        if self.network_dc_extrapolated is not None:
            return self.network_dc_extrapolated.frequency.f_scaled
        return None
    
    def get_dc_extrap_info(self):
        """Get DC extrapolation parameters that were set.
        
        Returns
        -------
        dict
            Parameters with non-None values
        """
        return {
            k: v for k, v in {
                'points': self.dc_extrap_points,
                'kind': self.dc_extrap_kind,
                'coords': self.dc_extrap_coords,
                'dc_sparam': 'Set' if self.dc_extrap_dc_sparam is not None else None
            }.items() if v is not None
        }
    
    # ==========================================
    # METADATA ACCESS
    # ==========================================
    
    def get_field_help(self, field_name: str) -> str:
        """Get help text for a field (useful for Qt tooltips).
        
        Parameters
        ----------
        field_name : str
            Name of the field
        
        Returns
        -------
        str
            Help text or empty string if not available
        """
        field = attrs.fields_dict(type(self)).get(field_name)
        if field and 'help' in field.metadata:
            return field.metadata['help']
        return ""
    
    def get_metadata_dict(self) -> dict:
        """Get all metadata as structured dictionary for Qt display.
        
        Returns
        -------
        dict
            Nested dict with categorized metadata
        """
        return {
            'File': {
                'Path': str(self.full_file_path),
                'Name': str(self.file_name),
                'Ports': self.num_ports,
                'Version': self.touchstone_version,
                'Format': self.touchstone_format,
            },
            'Instrument': {
                'Manufacturer': self.instrument_manufacturer,
                'Model': self.instrument_model,
                'Serial': self.instrument_serial,
                'Firmware': self.instrument_firmware,
            },
            'Frequency': {
                f'Start ({self.frequency_unit})': f"{self.start_freq_scaled:.6f}",
                f'Stop ({self.frequency_unit})': f"{self.stop_freq_scaled:.6f}",
                f'Center ({self.frequency_unit})': f"{self.center_freq_scaled:.6f}",
                f'Span ({self.frequency_unit})': f"{self.span_freq_scaled:.6f}",
                'Points': self.npoints,
                f'Step ({self.frequency_unit})': f"{self.freq_step_scaled:.6f}",
            },
            'VNA Config': {
                'IF BW (Hz)': self.if_bandwidth_hz,
                'Averages': self.num_averages,
                'Power (dBm)': self.source_power_dbm,
                'Ref Z (Ω)': self.reference_impedance,
            },
            'Calibration': {
                'Type': self.calibration_type,
                'Date': self.calibration_date,
                'Kit': self.calibration_kit,
            },
            'Time Domain': {
                'Resolution (ps)': f"{self.time_resolution_s*1e12:.3f}",
                'Max Range (ns)': f"{self.max_time_range_s*1e9:.3f}",
                'Velocity Factor': self.velocity_factor,
            },
        }
    
    def get_metadata_string(self, category=None) -> str:
        """Get formatted metadata string.
        
        Parameters
        ----------
        category : str, optional
            Category to display (None = all)
        
        Returns
        -------
        str
            Formatted metadata string
        """
        meta = self.get_metadata_dict()
        
        if category and category in meta:
            meta = {category: meta[category]}
        
        lines = []
        for cat_name, fields in meta.items():
            lines.append(f"\n{cat_name}:")
            lines.append("-" * 40)
            for field, value in fields.items():
                lines.append(f"  {field:20} : {value}")
        
        return '\n'.join(lines)
    
    # ==========================================
    # FILE I/O
    # ==========================================
    
    @classmethod
    def from_touchstone(cls, file_folder: Path, file_name: str, num_ports: int, **dc_kwargs):
        """Create TouchstoneData from file (explicit classmethod).
        
        Parameters
        ----------
        file_folder : Path
            Directory containing file
        file_name : str
            Filename without extension
        num_ports : int
            Number of ports
        **dc_kwargs
            DC extrapolation parameters (points, kind, coords, dc_sparam)
        
        Returns
        -------
        TouchstoneData
            Loaded instance
        """
        return cls(
            file_folder=file_folder,
            file_name=file_name,
            num_ports=num_ports,
            **dc_kwargs
        )
    
    def write_touchstone(self, output_path: Path, write_dc_extrap=False):
        """Write touchstone file with metadata preservation.
        
        Parameters
        ----------
        output_path : Path
            Output file path
        write_dc_extrap : bool
            Write DC-extrapolated network instead of original
        """
        network = self.network_dc_extrapolated if write_dc_extrap and self.network_dc_extrapolated else self.network_single_ended
        
        # Build metadata comment string
        meta_lines = [
            "! Touchstone file written by TouchstoneData",
            f"! File: {self.file_name}",
        ]
        
        if self.instrument_manufacturer:
            meta_lines.append(f"! VNA_MANUFACTURER: {self.instrument_manufacturer}")
        if self.instrument_model:
            meta_lines.append(f"! VNA_MODEL: {self.instrument_model}")
        if self.if_bandwidth_hz:
            meta_lines.append(f"! VNA_IFBW_HZ: {self.if_bandwidth_hz}")
        if self.velocity_factor != 1.0:
            meta_lines.append(f"! VELOCITY_FACTOR: {self.velocity_factor}")
        
        if self.comments:
            meta_lines.append("")
            meta_lines.append("! Original comments:")
            meta_lines.extend([f"! {line}" for line in self.comments.split('\n')])
        
        network.comments = '\n'.join(meta_lines)
        
        logger.info(f"Writing to: {output_path}")
        network.write_touchstone(str(output_path))
        logger.success("File written successfully")


# %% [markdown]
# ## Example Usage
#
# Demonstrations of key functionality.

# %%
def main():
    """Example usage demonstrating key features."""
    
    print("=" * 80)
    print("TOUCHSTONE HANDLER EXAMPLES")
    print("=" * 80)
    
    # Example 1: Load file
    print("\n1. LOAD 2-PORT FILE")
    print("-" * 80)
    data = TouchstoneData(
        file_folder=Path('.'),
        file_name='single_ended_2port',
        num_ports=2
    )
    print(data)
    
    # Example 2: Access frequency data
    print("\n2. FREQUENCY DATA")
    print("-" * 80)
    print(f"Frequency range: {data.start_freq_scaled:.3f} - {data.stop_freq_scaled:.3f} {data.frequency_unit}")
    print(f"Points: {data.npoints}")
    print(f"Step: {data.freq_step_scaled:.6f} {data.frequency_unit}")
    
    # Example 3: Time domain parameters
    print("\n3. TIME DOMAIN PARAMETERS")
    print("-" * 80)
    print(f"Resolution: {data.time_resolution_s*1e12:.3f} ps")
    print(f"Max range: {data.max_time_range_s*1e9:.3f} ns")
    print(f"DC extrap info: {data.get_dc_extrap_info()}")
    
    # Example 4: Modify metadata
    print("\n4. MODIFY METADATA")
    print("-" * 80)
    data.instrument_manufacturer = "Keysight"
    data.instrument_model = "N5232A"
    data.velocity_factor = 0.7
    print(f"Manufacturer: {data.instrument_manufacturer}")
    print(f"Model: {data.instrument_model}")
    print(f"Vp: {data.velocity_factor}")
    
    # Example 5: Regenerate DC extrapolation
    print("\n5. REGENERATE DC EXTRAPOLATION")
    print("-" * 80)
    data.generate_dc_extrapolation(points=50, kind='linear')
    print(f"Updated time resolution: {data.time_resolution_s*1e12:.3f} ps")
    print(f"Frequency comparison:")
    print(f"  Original: {len(data.frequency)} points")
    print(f"  DC extrap: {len(data.frequency_dc_extrap)} points")
    
    # Example 6: Metadata dictionary
    print("\n6. METADATA DICTIONARY (for Qt)")
    print("-" * 80)
    meta = data.get_metadata_dict()
    print(f"Categories: {list(meta.keys())}")
    
    # Example 7: Formatted metadata string
    print("\n7. FORMATTED METADATA")
    print("-" * 80)
    print(data.get_metadata_string(category='Frequency'))
    
    # Example 8: Write file
    print("\n8. WRITE FILE WITH METADATA")
    print("-" * 80)
    output_path = Path('.') / 'test_output.s2p'
    data.write_touchstone(output_path)
    print(f"Written to: {output_path}")
    
    # Example 9: Read back to verify
    print("\n9. READ BACK TO VERIFY")
    print("-" * 80)
    data_readback = TouchstoneData.from_touchstone(
        file_folder=Path('.'),
        file_name='test_output',
        num_ports=2
    )
    print(f"Manufacturer: {data_readback.instrument_manufacturer}")
    print(f"Model: {data_readback.instrument_model}")
    print(f"Vp: {data_readback.velocity_factor}")
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)


# %%
if __name__ == "__main__":
    main()