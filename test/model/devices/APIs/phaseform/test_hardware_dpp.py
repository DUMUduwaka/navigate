# Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted for academic and research use only (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holders nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
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

"""Hardware-in-the-loop tests for the Phaseform DPP.

These tests use a persistent device connection for the module and are
excluded from normal test runs by the ``hardware`` pytest marker.

Required environment variables:
    NAVIGATE_DPP_PORT
    NAVIGATE_DPP_CALIBRATION_FILE

Optional:
    NAVIGATE_DPP_OPERATION_MODE  -- "h" or "v" (default: "v")

Each test starts and ends with the device flattened, and the module fixture
flattens and disconnects the device during teardown.
"""

import os
import time

import numpy as np
import pytest

from navigate.model.devices.APIs.phaseform.phaseplate import PhaseformDPP

PORT = os.environ.get("NAVIGATE_DPP_PORT")
CALIBRATION_FILE = os.environ.get("NAVIGATE_DPP_CALIBRATION_FILE")
OPERATION_MODE = os.environ.get("NAVIGATE_DPP_OPERATION_MODE", "v")

SAFE_TEST_AMPLITUDE = 0.05
MODAL_APPROXIMATION_TOLERANCE = 0.02

pytestmark = pytest.mark.hardware


@pytest.fixture(scope="module")
def device():
    """Connect once for every test in this file, disconnect once at the end."""
    if PORT is None:
        pytest.skip(
            "NAVIGATE_DPP_PORT is not set. Real phaseform DPP hardware tests "
            "require an explicit serial port."
        )

    if CALIBRATION_FILE is None:
        pytest.skip(
            "NAVIGATE_DPP_CALIBRATION_FILE is not set. Real Phaseform DPP hardware"
            "tests require a calibration file."
        )

    if not os.path.isfile(CALIBRATION_FILE):
        pytest.skip(f"Phaseform calibration file does not exist {CALIBRATION_FILE}")

    if OPERATION_MODE not in ("v", "h"):
        pytest.fail(
            "NAVIGATE_DPP_OPERATION_MODE must be either 'h' or 'v', "
            f"got {OPERATION_MODE!r}."
        )

    time.sleep(2)
    dev = PhaseformDPP(
        port=PORT,
        calibration_file_path=CALIBRATION_FILE,
        operation_mode=OPERATION_MODE,
    )
    try:
        yield dev
    finally:
        try:
            dev.flat()
        finally:
            dev.disconnect()


@pytest.fixture
def plate(device):
    """Return a connected DPP flattened before and after each test."""
    device.flat()
    yield device
    device.flat()


def test_connect_with_load_default_loads_precalibration():
    """Verify load_default=True loads a usable default precalibration.

    Opens its own independent connection (a different load_default argument
    than the shared `device` fixture uses) -- must run before any test that
    triggers that module-scoped fixture, since a real serial port only
    allows one open connection at a time.
    """

    time.sleep(2)

    device = PhaseformDPP(
        port=PORT,
        calibration_file_path=CALIBRATION_FILE,
        operation_mode=OPERATION_MODE,
        load_default=True,
    )

    try:
        assert device.dpp.influence_matrix is not None
        assert device.dpp.corrections_loaded is True

    finally:
        device.flat()
        device.disconnect()


def test_connect_loads_a_real_influence_matrix(plate):
    assert plate.dpp.influence_matrix is not None
    assert isinstance(plate.dpp.influence_matrix, np.ndarray)
    rows, cols = plate.dpp.influence_matrix.shape
    assert rows > 1
    assert cols > 1


def test_connect_loads_flat_field_corrections(plate):
    """Verify connecting leaves flat-field corrections loaded."""
    assert plate.dpp.corrections_loaded is True


def test_flat_runs_without_error(plate):
    plate.flat()


def test_move_absolute_zero_runs_without_error(plate):
    plate.move_absolute_zero()


def test_flat_runs_after_zero_flatness(plate):
    plate.zero_flatness()
    plate.flat()


def test_display_modes_with_osa_dict_reaches_target_within_tolerance(plate):
    osa_index = 1  # vertical tilt

    plate.display_modes({osa_index: SAFE_TEST_AMPLITUDE})

    coefs, coef_inds = plate.get_modal_coefs()
    approximated = dict(zip(coef_inds, coefs))[osa_index]

    assert approximated == pytest.approx(
        SAFE_TEST_AMPLITUDE,
        abs=MODAL_APPROXIMATION_TOLERANCE,
    )


def test_display_modes_with_negative_amplitude_reaches_target_within_tolerance(plate):
    plate.display_modes({1: -SAFE_TEST_AMPLITUDE})

    coefs, coef_inds = plate.get_modal_coefs()
    approximated = dict(zip(coef_inds, coefs))[1]

    assert approximated == pytest.approx(
        -SAFE_TEST_AMPLITUDE,
        abs=MODAL_APPROXIMATION_TOLERANCE,
    )


def test_display_modes_with_multiple_modes_reaches_targets_within_tolerance(plate):
    """Verify multiple modes commanded together reach their targets within tolerance."""
    targets = {
        1: SAFE_TEST_AMPLITUDE,
        2: -SAFE_TEST_AMPLITUDE,
        4: SAFE_TEST_AMPLITUDE,
    }

    plate.display_modes(targets)

    coefs, coef_inds = plate.get_modal_coefs()
    approximated = dict(zip(coef_inds, coefs))

    for osa_index, target in targets.items():
        assert approximated[osa_index] == pytest.approx(
            target,
            abs=MODAL_APPROXIMATION_TOLERANCE,
        )


def test_get_modal_coefs_commanded_values_match_filtered_and_full_views(plate):
    """Verify commanded modes have identical readback in filtered and full views."""
    targets = {
        1: SAFE_TEST_AMPLITUDE,
        2: -SAFE_TEST_AMPLITUDE,
        4: SAFE_TEST_AMPLITUDE,
        6: -SAFE_TEST_AMPLITUDE,
    }

    plate.display_modes(targets)

    coefs_filtered, idx_filtered = plate.get_modal_coefs(only_commanded=True)
    coefs_full, idx_full = plate.get_modal_coefs(only_commanded=False)

    assert sorted(idx_filtered) == sorted(targets.keys())
    assert idx_full == list(range(1, plate.n_modes))

    approx_filtered = dict(zip(idx_filtered, coefs_filtered))
    approx_full = dict(zip(idx_full, coefs_full))

    for osa_index, target in targets.items():
        assert approx_filtered[osa_index] == approx_full[osa_index]
        assert approx_full[osa_index] == pytest.approx(
            target,
            abs=MODAL_APPROXIMATION_TOLERANCE,
        )


def test_display_modes_with_valid_length_list_reaches_target_within_tolerance(plate):
    coefs = [0.0] * (plate.n_modes - 1)
    coefs[0] = SAFE_TEST_AMPLITUDE  # first real mode

    plate.display_modes(coefs)

    approx_coefs, _ = plate.get_modal_coefs()

    assert approx_coefs[0] == pytest.approx(
        SAFE_TEST_AMPLITUDE,
        abs=MODAL_APPROXIMATION_TOLERANCE,
    )


def test_set_flat_with_zero_vector_returns_to_flat_baseline(plate):
    """Verify a zero flat vector leaves all non-piston modes near baseline."""
    n_terms = plate.dpp.influence_matrix.shape[0]

    plate.set_flat(pos=np.zeros(n_terms))
    plate.flat()  # set_flat() only updates the reference -- flat() applies it

    coefs, coef_inds = plate.get_modal_coefs(only_commanded=False)
    approximated = dict(zip(coef_inds, coefs))

    for value in approximated.values():
        assert abs(value) < MODAL_APPROXIMATION_TOLERANCE


def test_save_and_load_json_round_trip_restores_coefficients(plate, tmp_path):
    osa_index = 1
    plate.display_modes({osa_index: SAFE_TEST_AMPLITUDE})

    json_path = tmp_path / "hardware_test_coefs.json"
    plate.save_json(path=str(json_path))

    assert json_path.exists()

    plate.flat()
    loaded = plate.load_json(path=str(json_path))

    coefs_after_load, coef_inds_after_load = plate.get_modal_coefs()
    approximated = dict(zip(coef_inds_after_load, coefs_after_load))[osa_index]

    assert str(osa_index) in loaded["coefs"]
    assert approximated == pytest.approx(
        SAFE_TEST_AMPLITUDE,
        abs=MODAL_APPROXIMATION_TOLERANCE,
    )


def test_load_json_with_saved_voltages_restores_channel_voltages(plate, tmp_path):
    """Verify saved voltages are restored to the same actuator channels."""
    plate.display_modes({1: SAFE_TEST_AMPLITUDE})
    voltages_before = [int(v) for v in plate.dpp.voltages]

    json_path = tmp_path / "hardware_test_voltages.json"
    plate.save_json(path=str(json_path))

    assert json_path.exists()

    plate.flat()
    plate.load_json(
        path=str(json_path),
        use_saved_voltages=True,
    )

    assert list(plate.dpp.voltages)[1:] == voltages_before


def test_sequential_display_modes_calls_do_not_accumulate(plate):
    """Verify sequential display_modes() calls replace rather than accumulate."""
    plate.display_modes({1: SAFE_TEST_AMPLITUDE})
    plate.display_modes({1: -SAFE_TEST_AMPLITUDE})

    coefs, coef_inds = plate.get_modal_coefs()
    approximated = dict(zip(coef_inds, coefs))[1]

    assert approximated == pytest.approx(
        -SAFE_TEST_AMPLITUDE,
        abs=MODAL_APPROXIMATION_TOLERANCE,
    )


def test_flat_restores_modal_baseline_after_display_modes(plate):
    """Verify flat() returns a previously commanded mode to baseline."""
    osa_index = 1
    plate.display_modes({osa_index: SAFE_TEST_AMPLITUDE})

    plate.flat()

    coefs, coef_inds = plate.get_modal_coefs(only_commanded=False)
    approximated = dict(zip(coef_inds, coefs))[osa_index]

    assert approximated == pytest.approx(
        0.0,
        abs=MODAL_APPROXIMATION_TOLERANCE,
    )
