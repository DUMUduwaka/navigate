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

"""Hardware tests for the Phaseform DPP (the real plate on a serial port).

Run them only after the software tests pass, and only this file:

    set NAVIGATE_DPP_PORT=COM14
    set NAVIGATE_DPP_CALIBRATION_FILE=D:\\path\\to\\calibration.json
    python -m pytest test/model/devices/APIs/phaseform/test_hardware_dpp.py
        -o addopts= -m hardware -rs -v -s

They are skipped unless both variables are set, and excluded from normal runs by
the ``hardware`` marker. Close navigate and Phaseform's GUI first: only one
program can hold the port. Amplitudes are at most 0.05; the plate is flattened
before and after every test, and flattened and closed at the end.
"""

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.hardware

PORT = os.environ.get("NAVIGATE_DPP_PORT")
CALIBRATION = os.environ.get("NAVIGATE_DPP_CALIBRATION_FILE")
ORIENTATION = os.environ.get("NAVIGATE_DPP_OPERATION_MODE", "v")

AMPLITUDE = 0.05
#: |achieved - commanded| allowed for small commands (the SDK's model).
TOLERANCE = 0.02
DEFOCUS = 3  # navigate slot of OSA 4


@pytest.fixture(scope="module")
def plate():
    """One connection for the whole file; flattened and closed at the end."""
    if not PORT or not CALIBRATION:
        pytest.skip(
            "Set NAVIGATE_DPP_PORT and NAVIGATE_DPP_CALIBRATION_FILE to run the "
            "DPP hardware tests."
        )
    if not os.path.isfile(CALIBRATION):
        pytest.skip(f"Calibration file not found: {CALIBRATION}")

    from navigate.model.devices.APIs.phaseform.phaseplate import PhaseformDPP

    dpp = PhaseformDPP(
        port=PORT, calibration_file_path=CALIBRATION, operation_mode=ORIENTATION
    )
    try:
        yield dpp
    finally:
        try:
            dpp.flat()
        finally:
            dpp.disconnect()


@pytest.fixture
def flat_plate(plate):
    plate.flat()
    yield plate
    plate.flat()


def command(**slots):
    coefs = np.zeros(32, dtype=np.float32)
    for slot, value in slots.items():
        coefs[int(slot.lstrip("s"))] = value
    return coefs


def test_connects_with_the_calibration_max_voltage_and_limits(flat_plate):
    assert flat_plate.dpp.dpp_ctrl_opened
    # No max_voltage was requested: the calibration's "Max V" is kept.
    assert flat_plate.get_max_voltage() == flat_plate.dpp.maxV
    positive, negative = flat_plate.get_amplitude_limits()
    assert len(positive) == len(negative) == flat_plate.n_modes
    assert all(p > 0 > n for p, n in zip(positive, negative))
    print(
        f"max voltage {flat_plate.get_max_voltage()} V; defocus limits "
        f"+{positive[DEFOCUS]:.3f}/{negative[DEFOCUS]:.3f}"
    )


def test_flat_reads_back_zero_and_the_model_agrees(flat_plate):
    coefs, indices = flat_plate.get_modal_coefs()
    assert indices == list(range(1, 33))
    assert (coefs == 0).all()
    assert np.max(np.abs(flat_plate.get_achieved_modal_coefs())) < TOLERANCE


def test_small_defocus_is_reproduced(flat_plate):
    cmd = command(s3=AMPLITUDE)

    flat_plate.display_modes(cmd)

    np.testing.assert_array_equal(flat_plate.get_modal_coefs()[0], cmd)
    achieved = flat_plate.get_achieved_modal_coefs()
    assert achieved[DEFOCUS] == pytest.approx(AMPLITUDE, abs=TOLERANCE)
    assert np.max(np.abs(np.delete(achieved, DEFOCUS))) < TOLERANCE


def test_several_modes_together(flat_plate):
    cmd = command(s2=AMPLITUDE, s3=-AMPLITUDE, s4=AMPLITUDE, s6=-AMPLITUDE)

    flat_plate.display_modes(cmd)

    np.testing.assert_array_equal(flat_plate.get_modal_coefs()[0], cmd)
    np.testing.assert_allclose(
        flat_plate.get_achieved_modal_coefs(), cmd, atol=TOLERANCE
    )


def test_commands_are_absolute_not_added(flat_plate):
    flat_plate.display_modes(command(s3=AMPLITUDE))
    flat_plate.display_modes(command(s3=-AMPLITUDE))

    achieved = flat_plate.get_achieved_modal_coefs()
    assert achieved[DEFOCUS] == pytest.approx(-AMPLITUDE, abs=TOLERANCE)


def test_each_write_includes_the_settle_time(flat_plate):
    start = time.perf_counter()
    flat_plate.display_modes(command(s3=AMPLITUDE))
    elapsed = time.perf_counter() - start

    print(
        f"display_modes took {elapsed * 1000:.0f} ms "
        f"(settle {flat_plate.settle_time_ms} ms)"
    )
    assert flat_plate.settle_time_ms / 1000 <= elapsed < 1.0


def test_json_round_trip_restores_the_command(flat_plate, tmp_path):
    cmd = command(s3=AMPLITUDE, s4=-AMPLITUDE)
    flat_plate.display_modes(cmd)
    path = tmp_path / "dpp_round_trip.json"

    flat_plate.save_json(path=str(path))
    flat_plate.flat()
    loaded = flat_plate.load_json(path=str(path))

    np.testing.assert_allclose(loaded, cmd, atol=1e-6)
    np.testing.assert_allclose(flat_plate.get_modal_coefs()[0], cmd, atol=1e-6)


def test_reconnect_after_close(plate):
    """close() resets the plate; reconnecting needs a pause first."""
    plate.flat()
    plate.disconnect()
    assert not plate.dpp.dpp_ctrl_opened

    plate.display_modes(command(s3=AMPLITUDE))  # refused and logged: not connected
    assert (plate.get_modal_coefs()[0] == 0).all()

    waited = 0.0
    for pause in (2.0, 2.0, 4.0):
        time.sleep(pause)
        waited += pause
        try:
            plate.connect()
            break
        except ConnectionError:
            continue
    print(f"reconnected after {waited:.0f} s")
    assert plate.dpp.dpp_ctrl_opened
    plate.flat()
