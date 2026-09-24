"""
High-level controller for a Phaseform DPP phase plate
"""

import os
import json

import numpy as np
from dpp_ctrl import api_dpp
from dpp_ctrl.zernike_pol_calc import (
    get_classical_polynomial_name,
    get_orders_osa_index_correspondence,
    get_osa_standard_index,
)

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
    """

    def __init__(
        self,
        port,
        calibration_file_path,
        operation_mode="v",
        max_voltage=300,
        print_debug_info=False,
        load_default=False,
        connect=True,
        n_modes=36,
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
            Maximum drive voltage (100-300V), applied on every connect().

        print_debug_info : bool, optional
            Forwarded to PyCtrlDPP for verbose serial/debug logging.

        load_default : bool, optional
            If True, ignore calibration_file_path and load Phaseform's own
            bundled default calibration instead (PyCtrlDPP.load_precalibration()).

        connect : bool, optional
            If True (default), call connect() immediately during __init__.

        n_modes : int, optional
            Number of real (piston-excluded) Zernike modes
        """
        if not port:
            raise ValueError("Port must be provided.")
        if not calibration_file_path:
            raise ValueError("Calibration file path must be provided.")

        self.port = port
        self.calibration_file_path = calibration_file_path
        self.operation_mode = operation_mode
        self.max_voltage = max_voltage
        self.n_modes = n_modes
        self.load_default = load_default

        self.dpp = api_dpp.initialize(verbose_debug_info=print_debug_info)
        self._is_connected = False

        if connect:
            self.connect()

    def connect(self):
        """
        Open the serial connection (if not already connected) and (re)load
        calibration.
        """
        if not self._is_connected:
            try:
                connected = self.dpp.connect_device(port_name=self.port)
            except Exception as err:
                raise ConnectionError(
                    f"Could not open serial port '{self.port}' to connect to the DPP device. "
                    "Check that the device is powered on, connected via USB/serial."
                ) from err

            if not connected:
                raise ConnectionError(
                    f"Serial port '{self.port}' opened, but the DPP device did not respond. "
                    "Check that the device is not in use by another application and that the "
                    "correct port is specified."
                )
            self._is_connected = True
            print(f"Connected to DPP device on port '{self.port}'.")
            self.dpp.get_device_status()

        try:
            if self.max_voltage is not None:
                try:
                    self.dpp.set_max_voltage(self.max_voltage)
                except Exception as err:
                    raise RuntimeError(
                        f"Could not set max voltage to {self.max_voltage}V on the DPP device."
                    ) from err

                if self.dpp.max_voltage != self.max_voltage:
                    raise RuntimeError(
                        f"DPP rejected max voltage {self.max_voltage}V as out of its "
                        f"acceptable range [{self.dpp.minV}, {self.dpp.maxV}] and max "
                        f"voltage remains {self.dpp.max_voltage}V."
                    )

            if self.load_default:
                try:
                    calibration_files_loaded = self.dpp.load_precalibration(
                        operation_mode=self.operation_mode
                    )
                    print(
                        "Default calibrations files loaded:", calibration_files_loaded
                    )

                except Exception as err:
                    raise RuntimeError(
                        f"Could not load precalibration from "
                        f"'{self.operation_mode}'"
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
                    print("Influence matrix loaded: ", influence_matrix_loaded)

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
                        print(f"Flat offsets loaded: {flat_field_loaded}")

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
        except Exception:
            self.disconnect()
            raise

    def disconnect(self):
        """Disconnect from the DPP device."""
        if self._is_connected:
            self.dpp.close()
            self._is_connected = False

    def flat(self):
        """
        Apply the flat (zero-sum) Zernike profile. If the flat field correction
        has been loaded, it will be applied.

        """
        self.dpp.apply_flat_field()

    def zero_flatness(self):
        """
        Redefine flat as zero correction, then apply it.
        Phaseplate drives voltages to the all the channels.
        """
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
        """
        self.dpp.zero_outputs()

    def display_modes(self, coefs):
        """
        Command the DPP to reproduce the given Zernike modal coefficients.

        Parameters
        ----------

        coefs: list, np.ndarray or dict
            Dictionary or list with the specified Zernike polynomial coefficients to set on the device.
            Keys of a dict should be either integers (OSA index) or (m, n) order tuples.
            A list/array should NOT include a piston entry
                - index 0 should be your first real mode (e.g. vertical tilt).
                - This method prepends 0.0 for the piston slot automatically before calling
                  PyCtrlDPP.apply_phases(), which expects that slot to exist.
                - Length must be exactly self.n_modes - 1 (35 by default) --
                  matching mode_names/mode_orders' 35 real, piston-excluded
                  entries. A wrong length is rejected rather than silently
                  misapplied to the wrong modes.

        """
        if isinstance(coefs, dict):
            self.dpp.apply_phases(coefs)
            return

        coefs = np.asarray(coefs, dtype=np.float64)
        expected_len = self.n_modes - 1
        if coefs.shape[0] != expected_len:
            raise ValueError(
                f"display_modes: coefs length {coefs.shape[0]} does not match "
                f"the {expected_len} real Zernike modes expected (n_modes - 1)."
            )
        coefs = np.concatenate(([0.0], coefs))
        self.dpp.apply_phases(coefs)

    def get_modal_coefs(self, only_commanded=True):
        """
        Return the (coefficients, mode indices) actually approximated by
        the last-applied voltages. Piston value ignored.

        Parameters
        ----------
        only_commanded : bool, optional
            If True (default), return only the modes that were non-zero
            in the last display_modes()/apply_phases() call
            If False, return the full modal decomposition across all n_modes.
        """
        if only_commanded:
            approx = self.dpp.get_approx_ampls(get_all_approximated=False)
            if isinstance(approx, dict):
                entries = []
                for key, value in approx.items():
                    osa_idx = (
                        key if isinstance(key, int) else get_osa_standard_index(*key)
                    )
                    if osa_idx != 0:  # exclude piston
                        entries.append((osa_idx, value))
                entries.sort()
                coefs_idx = [i for i, _ in entries]
                coefs = [v for _, v in entries]
                return (coefs, coefs_idx)

            coefs_idx = [
                i
                for i in range(1, min(self.n_modes, len(approx)))
                if abs(approx[i]) > 0.0
            ]
            coefs = [approx[i] for i in coefs_idx]
            return (coefs, coefs_idx)

        coefs = self.dpp.get_approx_ampls(get_all_approximated=True)
        coefs = list(coefs[1 : self.n_modes])
        coefs_idx = list(range(1, len(coefs) + 1))  # exclude piston
        return (coefs, coefs_idx)

    def get_wavefront_pix(self):
        return

    def save_json(self, path=None, name=None):
        """
        Save the currently-approximated Zernike coefficients and raw
        per-channel voltages to a .json file, for later reuse.

        Parameters
        ----------

        path : str, optional
            Full path to write the .json file to.

        name : str, optional
            Alternatively, a name resolved to <calibration file's directory>/PhasePlate_files/<name>.json`.

        Either 'path' or 'name' must be provided.
        """

        if path:
            json_save_path = path
        elif name:
            save_dir = os.path.join(
                os.path.dirname(self.calibration_file_path), "PhasePlate_files"
            )
            os.makedirs(save_dir, exist_ok=True)
            json_save_path = os.path.join(save_dir, name + ".json")
        else:
            print("PhaseformDPP: Need to provide either name or path")
            return

        coefs, coefs_idx = self.get_modal_coefs()
        data = {
            "coefs": {str(i): c for i, c in zip(coefs_idx, coefs)},
            "voltages": self.dpp.voltages.tolist(),
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
            If False (default), reapply the saved Zernike coefficients via apply_phases() instead.

        Either 'path' or 'name' must be provided.

        Returns
        -------
        dict
            The loaded data, with both "coefs" and "voltages" keys.
        """
        if path:
            json_load_path = path
        elif name:
            load_dir = os.path.join(
                os.path.dirname(self.calibration_file_path), "PhasePlate_files"
            )
            json_load_path = os.path.join(load_dir, name + ".json")
        else:
            print("PhaseformDPP: Need to provide either name or path")
            return

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
            if use_saved_voltages:
                saved_voltages = data["voltages"]
                n_channels = self.dpp.influence_matrix.shape[1]
                pin_offset = 1 if len(saved_voltages) == n_channels else 0
                self.dpp.set_pins_volts(
                    {
                        int(pin) + pin_offset: int(volt)
                        for pin, volt in enumerate(saved_voltages)
                    }
                )

            else:
                coefs = {int(idx): float(coef) for idx, coef in data["coefs"].items()}
                self.dpp.apply_phases(coefs)

        except KeyError as err:
            raise ValueError(
                f"Json file {json_load_path} is missing expected key {err}."
            ) from err
        except TypeError as err:
            raise ValueError(
                f"Json file {json_load_path} does not have the expected "
                f"structure: {err}"
            ) from err

        return data
