# Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted for academic and research use only
# (subject to the limitations in the disclaimer below)
# provided that the following conditions are met:

#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.

#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.

#      * Neither the name of the copyright holders nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

# Standard Library Imports
import logging

# Third Party Imports

# Local Imports
from navigate.model.devices.mirror.base import MirrorBase
from navigate.model.devices.device_types import IntegratedDevice
from navigate.config.configuration_schema import SettingSpec
from navigate.model.devices.APIs.phaseform.phaseplate import PhaseformDPP
from navigate.tools.decorators import log_initialization

# Logger Setup
p = __name__.split(".")[1]
logger = logging.getLogger(p)


@log_initialization
class PhaseformDPPMirror(MirrorBase, IntegratedDevice):
    """PhaseformDPPMirror mirror class.

    Navigate treats the Phaseform DPP phase plate as a second manufacturer
    under the existing "mirror" device category (the same pattern the
    "camera" category already uses for multiple manufacturers), rather than
    as its own device category -- so this class inherits MirrorBase directly
    and reads the microscope's existing "mirror" config section, same as
    ImagineOpticsMirror.
    """

    configuration_schema = {
        "operation_mode": SettingSpec(
            str,
            default="v",
            label="Operation Mode",
            help_text=(
                "Orientation of the influence matrix to load from the "
                "calibration file."
            ),
            choices=("h", "v"),
            required=True,
        ),
    }

    @classmethod
    def get_connect_params(cls) -> list:
        """Connection parameters read from the mirror's `hardware` config block.

        Returns
        -------
        list
            Names of the parameters required to connect to the DPP.
        """
        return ["port", "calibration_file_path", "operation_mode"]

    @classmethod
    def connect(cls, port, calibration_file_path, operation_mode) -> PhaseformDPP:
        """Create and return a PhaseformDPP connection.

        Parameters
        ----------
        port : str
            Serial port the DPP is connected on (e.g. "COM4" or "/dev/ttyUSB0").
        calibration_file_path : str
            Path to the influence-matrix calibration file (.json/.csv/.mat).
            Contains both the influence matrix and the flat-field offsets.
        operation_mode : str
            'h'/'horizontal' or 'v'/'vertical' -- which orientation's
            influence matrix to load from the calibration file.

        Returns
        -------
        PhaseformDPP
            The mirror controller instance.
        """
        return PhaseformDPP(
            port=port,
            calibration_file_path=calibration_file_path,
            operation_mode=operation_mode,
        )

    def __init__(
        self, microscope_name, device_connection, configuration, *args, **kwargs
    ):
        """Initialize the PhaseformDPPMirror class.

        Parameters
        ----------
        microscope_name : str
            Name of the microscope.
        device_connection : PhaseformDPP
            The cached PhaseformDPP connection instance from the factory.
        configuration : dict
            Dictionary containing the configuration information.
        """
        super().__init__(microscope_name, device_connection, configuration)

        # obj: mirror controller (PhaseformDPP)
        self.mirror_controller = device_connection

        # PhaseformDPP.n_modes is piston-inclusive (display_modes() expects
        # exactly n_modes - 1 real coefficients). Adaptive-optics correction
        # always works with 32 real modes, so fix n_modes at 33 here (32
        # real + 1 piston slot) regardless of the device's own 36-mode
        # default, so display_modes()/get_modal_coefs() work in terms of
        # exactly 32 real coefficients.
        self.mirror_controller.n_modes = 33

        logger.info("PhaseformDPPMirror Initialized")

        # flatten the mirror
        self.flat()

    def __del__(self) -> None:
        """Delete the PhaseformDPPMirror class."""
        pass

    def flat(self):
        """Move the mirror to the flat position."""
        self.mirror_controller.flat()

    def zero_flatness(self):
        """Zero the mirror flatness."""
        self.mirror_controller.move_absolute_zero()

    def set_positions_flat(self, pos):
        """Set the mirror to the flat position.

        Parameters
        ----------
        pos : list
            List of Zernike coefficients to use as the new flat reference.
        """
        self.mirror_controller.set_flat(pos)

    def display_modes(self, coefs):
        """Display the mirror modes.

        Parameters
        ----------
        coefs : list or dict
            Zernike coefficients to display on the mirror.
        """
        self.mirror_controller.display_modes(coefs)

    def get_modal_coefs(self):
        """Get the modal coefficients of the mirror.

        Uses only_commanded=False so the result is a dense,
        positionally-ordered array (like ImagineOpticsMirror's), rather than
        PhaseformDPP's default sparse (non-zero modes only) readout --
        callers that drive both manufacturers generically (e.g. TonyWilson's
        convergence check) need a consistent, comparable shape.

        Returns
        -------
        tuple
            (coefficients, OSA indices) currently approximated on the
            device, dense across all 32 real modes.
        """
        return self.mirror_controller.get_modal_coefs(only_commanded=False)

    def set_from_json_file(self, path=None, name=None):
        """Load a saved Zernike coefficient file and apply it.

        Parameters
        ----------
        path : str, optional
            Path to the coefficient file, by default None
        name : str, optional
            Name of the coefficient file, by default None

        Returns
        -------
        dict
            The Zernike coefficients that were applied, keyed by OSA index.
        """
        if path:
            data = self.mirror_controller.load_json(path=path)
        elif name:
            data = self.mirror_controller.load_json(name=name)
        else:
            raise ValueError("Either 'path' or 'name' must be provided.")

        return {int(idx): float(coef) for idx, coef in data["coefs"].items()}

    def save_json_file(self, path=None, name=None):
        """Save the currently-applied Zernike coefficients to file.

        Parameters
        ----------
        path : str, optional
            Path to save the coefficient file, by default None
        name : str, optional
            Name of the coefficient file, by default None
        """
        if path:
            self.mirror_controller.save_json(path=path)
        elif name:
            self.mirror_controller.save_json(name=name)
        else:
            raise ValueError("Either 'path' or 'name' must be provided.")
