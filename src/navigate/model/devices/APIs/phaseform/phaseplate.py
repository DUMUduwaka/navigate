"""
High-level controller for a Phaseform DPP phase plate
"""

import contextlib
import io
import json
import logging
import os
import threading
import time

import numpy as np

try:
    from dpp_ctrl import api_dpp
    from dpp_ctrl.zernike_pol_calc import (
        get_classical_polynomial_name,
        get_orders_osa_index_correspondence,
    )
except ImportError as err:
    raise ImportError(
        "dpp_ctrl (the Phaseform DPP control library) is not installed in this "
        "Python environment. Install the dpp_ctrl package supplied by Phaseform "
        "into the environment that runs navigate."
    ) from err

# Logger Setup
p = __name__.split(".")[1]
logger = logging.getLogger(p)

#: float: |achieved - commanded| (calibration units) above which a command is
#: reported as not reproduced by the plate (saturation, or an unformable shape).
ACHIEVED_TOLERANCE = 0.02

#: float: minimum seconds between repeated warnings of the same kind.
WARNING_INTERVAL_S = 5.0

# Mode names are not in order that is compatible with the IMOP_Mirror class.
# Define the max radial order of Zernike polynomials to include in the list of mode names.
# Radial order 7, 36 Zernike modes.
radial_order = 7
mode_orders = get_orders_osa_index_correspondence(radial_order)
mode_names = [
    get_classical_polynomial_name(mode, short_names=True) for mode in mode_orders
]


class PhaseformDPP:
    """
    High-level controller for a Phaseform DPP phase plate.

    Navigate coefficient slot i (0-based) is the Phaseform OSA index i + 1
    (piston, OSA 0, is never commanded). Values are passed to the SDK in its
    own units, unchanged.
    """

    def __init__(
        self,
        port,
        calibration_file_path,
        operation_mode="v",
        max_voltage=None,
        print_debug_info=False,
        load_default=False,
        connect=True,
        n_modes=32,
        settle_time_ms=65,
    ):
        """
        Connect to the DPP and load its influence-matrix/flat-field calibration.

        Parameters
        ----------

        port : str
            Serial port the DPP is connected on (e.g. "COM4" or "/dev/ttyUSB0").

        calibration_file_path : str
            Path to the influence-matrix calibration file (.json/.csv/.mat).
            This contains both influence matrix and the flat field offsets.

        operation_mode : str, optional
            'h'/'horizontal' or 'v'/'vertical' -- which orientation's
            influence matrix to load from the calibration file. Default "v".

        max_voltage : int, optional
            Maximum drive voltage in V, applied after the calibration on every
            connect(). None (default) keeps the maximum the calibration file
            sets when it loads. A value outside the range the calibration
            allows raises.

        print_debug_info : bool, optional
            Forwarded to PyCtrlDPP for verbose serial/debug logging.

        load_default : bool, optional
            If True, ignore calibration_file_path and load Phaseform's own
            bundled default calibration instead (PyCtrlDPP.load_precalibration()).

        connect : bool, optional
            If True (default), call connect() immediately during __init__.

        n_modes : int, optional
            Number of real Zernike modes (piston excluded) navigate commands.
            Slot i maps to OSA index i + 1. Default 32.

        settle_time_ms : int, optional
            Time to wait after each display_modes() write before returning, so
            the next camera exposure sees the new shape. The SDK returns as
            soon as the bytes are handed to the serial driver. Default 65 ms
            (what Phaseform's own tests wait); 0-1000.
        """
        if not port:
            raise ValueError("Port must be provided.")
        if not calibration_file_path:
            raise ValueError("Calibration file path must be provided.")

        self.port = port
        self.calibration_file_path = calibration_file_path
        self.operation_mode = operation_mode
        self.max_voltage = max_voltage
        self.n_modes = int(n_modes)
        self.load_default = load_default

        self.dpp = api_dpp.initialize(verbose_debug_info=print_debug_info)
        # Always solve fresh: the SDK's volts cache is keyed only by device ID,
        # orientation and amplitudes (not max_voltage or calibration content),
        # and it writes a pickle into site-packages. Must be off before loading.
        self.dpp.save_volts_flag = False
        self._is_connected = False
        self.set_settle_time(settle_time_ms)

        #: dict: time of the last warning of each kind (rate limiting).
        self._last_warning = {}

        if not 1 <= self.n_modes <= self.dpp.max_osa_index:
            raise ValueError(
                f"n_modes must be between 1 and {self.dpp.max_osa_index}, "
                f"got {self.n_modes}."
            )

        #: threading.RLock: serializes every SDK call (signal and data threads).
        self._lock = threading.RLock()

        #: np.ndarray: last successfully commanded coefficients (navigate slots).
        self._commanded = np.zeros(self.n_modes, dtype=np.float32)

        #: tuple: per-slot (positive, negative) amplitude limits, or (None, None).
        self._limits = (None, None)

        if connect:
            self.connect()

    def connect(self):
        """
        Load and check the calibration and the max voltage, compute the
        amplitude limits (none of which needs the port), then open the serial
        port if it is not open yet. Configuration errors therefore fail before
        the port is opened and never reset the plate. (With load_default, the
        port is opened first, as before.)
        """
        with self._lock:
            try:
                if self.load_default and not self._is_connected:
                    # Phaseform's bundled calibration may depend on the device.
                    self._open_port()
                self._load_calibration()
                self._compute_amplitude_limits()
                if not self._is_connected:
                    self._open_port()
            except Exception as err:
                # auto_redial only logs retries at DEBUG level, so report the
                # real cause here (closing an open port resets the plate).
                logger.error(f"PhaseformDPP: setup on port '{self.port}' failed: {err}")
                self.disconnect()
                raise

    def _open_port(self):
        """Open the serial port and query the device status.

        The SDK reports connection problems only by printing, so its output is
        captured, logged, and included in the error.
        """
        try:
            connected, output = self._call_logging_output(
                self.dpp.connect_device, port_name=self.port
            )
        except Exception as err:
            raise ConnectionError(
                f"Could not open serial port '{self.port}' for the DPP. Check that "
                "the device is powered on and connected."
            ) from err

        if not connected:
            raise ConnectionError(
                f"The DPP did not connect on '{self.port}'. Check the port name, "
                "that the device is powered on, and that no other program (e.g. "
                f"Phaseform's GUI) is using it. SDK output: {self._one_line(output)}"
            )
        self._is_connected = True
        logger.info(f"Connected to DPP device on port '{self.port}'.")
        self._call_logging_output(self.dpp.get_device_status)

    @staticmethod
    def _one_line(text, limit=500):
        """Collapse multi-line SDK output for an error message."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return (" | ".join(lines)[-limit:]) if lines else "none"

    @staticmethod
    def _call_logging_output(func, *args, **kwargs):
        """Call an SDK function and log what it prints to stdout.

        Returns
        -------
        tuple
            (return value, printed text)
        """
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                result = func(*args, **kwargs)
        finally:
            output = buffer.getvalue().strip()
            if output:
                logger.info(f"PhaseformDPP SDK output:\n{output}")
        return result, output

    def _load_calibration(self):
        """Load the influence matrix and flat offsets, then the max voltage.

        Works without an open port. Raises RuntimeError if anything is invalid.
        """
        if self.load_default:
            try:
                calibration_files_loaded = self.dpp.load_precalibration(
                    operation_mode=self.operation_mode
                )
                logger.info(
                    f"Default calibrations files loaded: {calibration_files_loaded}"
                )

            except Exception as err:
                raise RuntimeError(
                    f"Could not load precalibration from '{self.operation_mode}'"
                ) from err

            if not calibration_files_loaded:
                raise RuntimeError(
                    f"PyCtrlDPP.load_precalibration(operation_mode = '{self.operation_mode}') "
                    "returned False and no valid default calibration was loaded."
                )

        else:
            try:
                influence_matrix_loaded = self.dpp.load_infl_matrix(
                    abs_path=self.calibration_file_path,
                    operation_mode=self.operation_mode,
                )
                logger.info(f"Influence matrix loaded: {influence_matrix_loaded}")

            except Exception as err:
                raise RuntimeError(
                    f"Could not load influence matrix calibration from "
                    f"'{self.calibration_file_path}' (orientation '{self.operation_mode}')"
                ) from err

            if not influence_matrix_loaded:
                raise RuntimeError(
                    f"PyCtrlDPP.load_infl_matrix('{self.calibration_file_path}', "
                    f"operation_mode = '{self.operation_mode}') returned False. "
                    "No exception was raised, but the file wasn't loaded as a valid "
                    "influence matrix."
                )

            if not self.dpp.corrections_loaded:
                try:
                    flat_field_loaded = self.dpp.load_flat_field(
                        abs_path=self.calibration_file_path,
                        operation_mode=self.operation_mode,
                    )
                    logger.info(f"Flat offsets loaded: {flat_field_loaded}")

                except Exception as err:
                    raise RuntimeError(
                        f"Could not load flat field offsets from "
                        f"'{self.calibration_file_path}' (orientation "
                        f"'{self.operation_mode}')."
                    ) from err

                if not flat_field_loaded:
                    raise RuntimeError(
                        f"PyCtrlDPP.load_flat_field(abs_path='{self.calibration_file_path}', "
                        f"operation_mode='{self.operation_mode}') returned False. No exception "
                        "was raised but no valid flat field offsets were loaded. "
                    )

        # Loading an influence matrix sets the SDK's max voltage (and its
        # allowed range) from the calibration file, so a requested max voltage
        # is applied after it. None keeps the calibration's value.
        if self.max_voltage is not None:
            self._apply_max_voltage(self.max_voltage)

        logger.info(
            f"PhaseformDPP max voltage: {self.dpp.max_voltage} V "
            f"(calibration allows {self.dpp.minV}-{self.dpp.maxV} V)."
        )

    def _apply_max_voltage(self, max_voltage):
        """Set the SDK max voltage, raising if the SDK silently rejects it."""
        try:
            self.dpp.set_max_voltage(max_voltage)
        except Exception as err:
            raise RuntimeError(
                f"Could not set max voltage to {max_voltage}V on the DPP device."
            ) from err

        if self.dpp.max_voltage != max_voltage:
            raise RuntimeError(
                f"DPP rejected max voltage {max_voltage}V as out of its "
                f"acceptable range [{self.dpp.minV}, {self.dpp.maxV}] and max "
                f"voltage remains {self.dpp.max_voltage}V."
            )

    def set_max_voltage(self, max_voltage):
        """Change the max drive voltage and recompute the amplitude limits.

        Parameters
        ----------
        max_voltage : int
            Maximum drive voltage (within the device's accepted range).
        """
        with self._lock:
            self._apply_max_voltage(int(max_voltage))
            self.max_voltage = int(max_voltage)
            self._compute_amplitude_limits()

    def get_max_voltage(self):
        """Return the max drive voltage the SDK is actually using (V)."""
        return self.dpp.max_voltage

    def set_settle_time(self, settle_time_ms):
        """Set the wait after each display_modes() write.

        Parameters
        ----------
        settle_time_ms : int
            Milliseconds, 0-1000.
        """
        settle_time_ms = int(settle_time_ms)
        if not 0 <= settle_time_ms <= 1000:
            raise ValueError(
                f"settle_time_ms must be between 0 and 1000, got {settle_time_ms}."
            )
        self.settle_time_ms = settle_time_ms

    def _ready_to_write(self, action):
        """Return True if the port is open; otherwise log that nothing is sent.

        The SDK itself only prints (and returns) when the device is not open.
        """
        if self._is_connected and getattr(self.dpp, "dpp_ctrl_opened", False):
            return True
        logger.error(
            f"PhaseformDPP.{action}: the device is not connected; nothing was sent."
        )
        return False

    def _sdk_accepted(self, coefs):
        """Return True if the SDK parsed exactly these amplitudes (OSA 1..n)."""
        parsed = np.asarray(getattr(self.dpp, "zernike_amplitudes", ()), dtype=float)
        if parsed.shape[0] < self.n_modes + 1:
            return False
        return np.allclose(parsed[1 : self.n_modes + 1], coefs, rtol=0, atol=1e-9)

    def _all_zero_voltages(self):
        """Return True if the SDK's last solved voltages are all zero."""
        volts = np.asarray(getattr(self.dpp, "voltages", ()), dtype=float).ravel()
        return volts.size > 0 and not volts.any()

    def _warn_throttled(self, kind, message):
        """Log a warning at most once per WARNING_INTERVAL_S for each kind."""
        now = time.monotonic()
        last = self._last_warning.get(kind)
        if last is None or now - last >= WARNING_INTERVAL_S:
            self._last_warning[kind] = now
            logger.warning(message)

    def _compute_amplitude_limits(self):
        """Ask the SDK for the per-mode achievable range (calculation only).

        A failure is logged and leaves the limits unset; it never fails connect().
        """
        with self._lock:
            self._limits = (None, None)
            try:
                computed = self.dpp.get_ampls_limits()
            except Exception:
                logger.warning(
                    "PhaseformDPP: could not compute amplitude limits.", exc_info=True
                )
                return
            # get_ampls_limits() reuses the SDK's solver state (calculate-only);
            # recompute the last command so achieved values stay meaningful.
            try:
                self.dpp.apply_phases(
                    [0.0] + [float(c) for c in self._commanded], calculate_only=True
                )
            except Exception:
                logger.warning(
                    "PhaseformDPP: could not restore solver state.", exc_info=True
                )
            pos = list(getattr(self.dpp, "ampls_limits_pos_list", []) or [])
            neg = list(getattr(self.dpp, "ampls_limits_neg_list", []) or [])
            if (
                not computed
                or len(pos) < self.n_modes + 1
                or len(neg) < self.n_modes + 1
            ):
                logger.warning("PhaseformDPP: amplitude limits are not available.")
                return
            self._limits = (
                [float(v) for v in pos[1 : self.n_modes + 1]],
                [float(v) for v in neg[1 : self.n_modes + 1]],
            )
            logger.info(
                "PhaseformDPP amplitude limits (slot: +max/-max): "
                + ", ".join(
                    f"{i}: +{a:.3f}/{b:.3f}"
                    for i, (a, b) in enumerate(zip(*self._limits))
                )
            )

    def get_amplitude_limits(self):
        """
        Per-mode achievable +/- amplitude, computed at connect() time for each
        mode on its own (other modes zero), relative to flat.

        Returns
        -------
        tuple(list[float], list[float]) or (None, None)
            (positive_limits, negative_limits), each of length n_modes and
            aligned with navigate's coefficient slots, or (None, None).
        """
        pos, neg = self._limits
        if pos is None:
            return (None, None)
        return (list(pos), list(neg))

    def disconnect(self):
        """Disconnect from the DPP device, including a port left open by a
        connect_device() that failed after opening it."""
        with self._lock:
            if (
                self._is_connected
                or getattr(self.dpp, "device_handle", None) is not None
            ):
                self.dpp.close()
            self._is_connected = False
            self.dpp.dpp_ctrl_opened = False  # the SDK's close() leaves it True

    def flat(self):
        """
        Apply the flat (zero-sum) Zernike profile. If the flat field correction
        has been loaded, it will be applied. Errors are logged, not raised.
        """
        with self._lock:
            if not self._ready_to_write("flat"):
                return
            try:
                self.dpp.apply_flat_field()
            except Exception:
                logger.exception("PhaseformDPP: apply_flat_field failed.")
                return
            self._commanded = np.zeros(self.n_modes, dtype=np.float32)

    def zero_flatness(self):
        """
        Redefine flat as zero correction, then apply it.
        Phaseplate drives voltages to the all the channels.
        """
        with self._lock:
            self.set_flat(pos=np.zeros(self.dpp.influence_matrix.shape[0]))
            self.flat()

    def set_flat(self, pos=None, pos_path=None):
        """
        Redefine the flat field reference used by flat() without moving the device

        Parameters
        ----------

        pos: list or np.ndarray
            Zernike coefs per mode for flat-field correction. Length must match the number of Zernike terms
            in the loaded influence matrix.

        pos_path: str
            Alternatively, a calibration *.json/*.csv/*.mat file to load
            flat field offsets from.

        Either pos or pos_path must be provided. If both are given, pos_path
        takes precedence and pos is silently ignored.
        """
        with self._lock:
            if pos_path:
                if not self.dpp.load_flat_field(
                    abs_path=pos_path, operation_mode=self.operation_mode
                ):
                    raise RuntimeError(
                        f"Could not load flat field reference from '{pos_path}'"
                    )
                return

            if pos is None:
                raise ValueError(
                    "PhaseformDPP.set_flat: provide either 'pos' or 'pos_path'."
                )

            pos = np.asarray(pos, dtype=np.float64)
            if pos.shape[0] != self.dpp.influence_matrix.shape[0]:
                raise ValueError(
                    f"Flat reference length {pos.shape[0]} does not match the "
                    f"{self.dpp.influence_matrix.shape[0]} Zernike modes defined by the "
                    "loaded influence matrix."
                )

            self.dpp.flatten_field_coefficients = pos
            self.dpp.corrections_loaded = True

    def move_absolute_zero(self):
        """
        Zero all 64 channels directly on hardware, bypassing the influence matrix solve entirely.
        Errors are logged, not raised.
        """
        with self._lock:
            if not self._ready_to_write("move_absolute_zero"):
                return
            try:
                self.dpp.zero_outputs()
            except Exception:
                logger.exception("PhaseformDPP: zero_outputs failed.")

    def display_modes(self, coefs):
        """
        Command the DPP to reproduce the given Zernike modal coefficients.

        Parameters
        ----------

        coefs: list, np.ndarray or dict
            A list/array of exactly n_modes values: slot i is OSA index i + 1
            (no piston entry). A dict maps OSA index (1..n_modes) to value;
            missing modes are 0.

        The command is absolute (not added to the previous one). Errors are
        logged, not raised; on error get_modal_coefs() keeps the previous
        command.
        """
        if isinstance(coefs, dict):
            dense = np.zeros(self.n_modes, dtype=np.float64)
            for key, value in coefs.items():
                idx = int(key)
                if 1 <= idx <= self.n_modes:
                    dense[idx - 1] = float(value)
                else:
                    logger.warning(
                        f"PhaseformDPP.display_modes: OSA index {idx} ignored "
                        f"(valid 1..{self.n_modes})."
                    )
            coefs = dense

        try:
            coefs = np.asarray(coefs, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            logger.error("PhaseformDPP.display_modes: coefficients are not numeric.")
            return

        if coefs.shape[0] != self.n_modes:
            logger.error(
                f"PhaseformDPP.display_modes: got {coefs.shape[0]} coefficients, "
                f"expected {self.n_modes}."
            )
            return

        if not np.isfinite(coefs).all():
            logger.error(
                "PhaseformDPP.display_modes: coefficients contain NaN or inf; "
                "nothing was sent."
            )
            return

        commanded = coefs.astype(np.float32)
        # Python floats only: the SDK silently truncates a list at the first
        # element that is not a Python float/int (e.g. np.float32).
        phases = [0.0] + [float(c) for c in coefs]

        with self._lock:
            if not self._ready_to_write("display_modes"):
                return
            try:
                self.dpp.apply_phases(phases)
            except Exception:
                logger.exception("PhaseformDPP: apply_phases failed.")
                return
            # The SDK returns silently (it only prints) when it cannot parse the
            # phases (e.g. a list cut short at an np.float32); then the parsed
            # amplitudes differ from the command and the readback must not change.
            if not self._sdk_accepted(coefs):
                logger.error(
                    "PhaseformDPP.display_modes: the SDK did not accept the "
                    "command; nothing was sent."
                )
                return
            # A failed voltage solve makes the SDK send 0 V to every channel
            # without an error; the plate is then not at the command.
            if self._all_zero_voltages():
                logger.error(
                    "PhaseformDPP.display_modes: the SDK's voltage solve failed "
                    "(all channels 0 V); the plate is not at the command."
                )
                return
            self._commanded = commanded
            if self.settle_time_ms:
                time.sleep(self.settle_time_ms / 1000)
            self._check_command(commanded)

    def _check_command(self, commanded):
        """Warn (rate-limited) when a command is outside the single-mode limits
        or the model predicts the plate does not reproduce it."""
        pos, neg = self._limits
        if pos is not None:
            outside = [i for i, c in enumerate(commanded) if c > pos[i] or c < neg[i]]
            if outside:
                self._warn_throttled(
                    "limits",
                    "PhaseformDPP: commanded value outside the achievable range "
                    f"for slot(s) {outside}; the plate will saturate.",
                )
        achieved = self.get_achieved_modal_coefs()
        mismatch = np.abs(achieved - commanded)
        worst = float(np.max(mismatch))
        if worst > ACHIEVED_TOLERANCE:
            slots = [i for i, d in enumerate(mismatch) if d > ACHIEVED_TOLERANCE]
            self._warn_throttled(
                "achieved",
                f"PhaseformDPP: the plate reproduces the command only to within "
                f"{worst:.3f} (slot(s) {slots}): saturation, or a shape it cannot "
                "form.",
            )
        else:
            logger.debug(f"PhaseformDPP: max |achieved - commanded| = {worst:.4f}")

    def get_modal_coefs(self):
        """
        Return (coefficients, mode indices) last commanded via display_modes()
        or flat(), like IMOP_Mirror.get_modal_coefs(). Indices are 1..n_modes
        (OSA index of each slot).
        """
        return (self._commanded.copy(), list(range(1, self.n_modes + 1)))

    def get_achieved_modal_coefs(self):
        """
        Return the coefficients the calibration model predicts for the voltages
        last calculated (a model prediction, not a measurement), aligned with
        navigate's slots.
        """
        with self._lock:
            approx = self.dpp.get_approx_ampls(get_all_approximated=True)
        approx = np.asarray(approx, dtype=np.float64).ravel()
        if approx.shape[0] < self.n_modes + 1:
            return np.zeros(self.n_modes, dtype=np.float64)
        return approx[1 : self.n_modes + 1].copy()

    def get_wavefront_pix(self):
        return

    def _resolve_path(self, path, name):
        """Return the .json path from 'path' or 'name', or None."""
        if path:
            return path
        if name:
            files_dir = os.path.join(
                os.path.dirname(self.calibration_file_path), "PhasePlate_files"
            )
            os.makedirs(files_dir, exist_ok=True)
            return os.path.join(files_dir, name + ".json")
        logger.error("PhaseformDPP: Need to provide either name or path")
        return None

    def save_json(self, path=None, name=None):
        """
        Save the last commanded coefficients, the model-predicted (achieved)
        coefficients and the per-channel voltages to a .json file.

        Parameters
        ----------

        path : str, optional
            Full path to write the .json file to.

        name : str, optional
            Alternatively, a name resolved to <calibration file's directory>/PhasePlate_files/<name>.json`.

        Either 'path' or 'name' must be provided.
        """
        json_save_path = self._resolve_path(path, name)
        if json_save_path is None:
            return

        with self._lock:
            commanded = self._commanded.copy()
            achieved = self.get_achieved_modal_coefs()
            voltages = np.asarray(getattr(self.dpp, "voltages", []), dtype=float)

        data = {
            "coefs": {str(i + 1): float(c) for i, c in enumerate(commanded)},
            "achieved_coefs": {str(i + 1): float(c) for i, c in enumerate(achieved)},
            "voltages": voltages.tolist(),
        }

        with open(json_save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path=None, name=None, use_saved_voltages=False):
        """
        Load a previously-saved correction (from save_json()) and apply it
        to the device.

        Parameters
        ----------

        path : str, optional
            Full path to the .json file to load.

        name : str, optional
            Alternatively, a name resolved to <calibration file's directory>/PhasePlate_files/<name>.json`.

        use_saved_voltages : bool, optional
            If True, reapply the exact saved per-channel voltages directly.
            If False (default), reapply the saved coefficients via display_modes().

        Either 'path' or 'name' must be provided.

        Returns
        -------
        list
            n_modes coefficients (slot i = OSA index i + 1; missing modes 0.0),
            or None if neither path nor name was given.
        """
        json_load_path = self._resolve_path(path, name)
        if json_load_path is None:
            return None

        try:
            with open(json_load_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"Json file {json_load_path} does not exist."
            ) from err
        except json.JSONDecodeError as err:
            raise ValueError(f"Json file {json_load_path} is not valid JSON.") from err

        try:
            coefs = [0.0] * self.n_modes
            for key, value in data["coefs"].items():
                idx = int(key)
                if 1 <= idx <= self.n_modes:
                    coefs[idx - 1] = float(value)
                else:
                    logger.warning(
                        f"PhaseformDPP.load_json: OSA index {idx} in "
                        f"{json_load_path} ignored (valid 1..{self.n_modes})."
                    )

            if use_saved_voltages:
                saved_voltages = data["voltages"]
                with self._lock:
                    if not self._ready_to_write("load_json"):
                        return None
                    n_channels = self.dpp.influence_matrix.shape[1]
                    pin_offset = 1 if len(saved_voltages) == n_channels else 0
                    self.dpp.set_pins_volts(
                        {
                            int(pin) + pin_offset: int(volt)
                            for pin, volt in enumerate(saved_voltages)
                        }
                    )
                    self._commanded = np.asarray(coefs, dtype=np.float32)
            else:
                self.display_modes(coefs)

        except KeyError as err:
            raise ValueError(
                f"Json file {json_load_path} is missing expected key {err}."
            ) from err
        except (TypeError, ValueError, AttributeError) as err:
            raise ValueError(
                f"Json file {json_load_path} does not have the expected "
                f"structure: {err}"
            ) from err

        return coefs
