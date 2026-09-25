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

"""Wiring + logic tests for navigate's PhaseformDPP (APIs/phaseform/phaseplate.py):
a fake PyCtrlDPP stands in for the real serial-connected device (shapes/
attributes/methods matched against the real dpp_ctrl.api_dpp.PyCtrlDPP -- see
test_dpp_sdk_contract.py for tests that verify those assumptions against the
real library), so these confirm PhaseformDPP's own logic -- argument wiring,
shape/length validation, piston handling, error handling -- without needing
dpp_ctrl to actually open a serial port or load real calibration files.

Ported from the standalone PhaseformDPPController repo's
tests/test_phaseform_dpp.py, which tests an identical PhaseformDPP class body
-- only the import path changes.
"""

import json

import numpy as np
import pytest

from navigate.model.devices.APIs.phaseform.phaseplate import (
    PhaseformDPP,
    mode_names,
    mode_orders,
)

N_TERMS = 91
N_CHANNELS = 63
N_RAW_VOLTAGE_SLOTS = 64


class FakePyCtrlDPP:
    """Stands in for dpp_ctrl.api_dpp.PyCtrlDPP. Method signatures, default
    argument names, and attribute shapes are matched against the real class
    (test_api_dpp_contract.py checks that match still holds)."""

    def __init__(self, print_debug_info=False):
        self.calls = []
        self.print_debug_info = print_debug_info
        self.connected = True
        self.influence_matrix = np.zeros((N_TERMS, N_CHANNELS))
        self.flatten_field_coefficients = None
        self.corrections_loaded = False
        self.default_calibrations_path = None
        self.voltages = np.zeros(shape=N_RAW_VOLTAGE_SLOTS, dtype="uint16")
        self.max_osa_index = N_TERMS - 1
        self._approx_ampls = [0.0] + [round(0.1 * i, 4) for i in range(1, N_TERMS)]
        # Confirmed against the real library in test_api_dpp_contract.py:
        # minV=50, maxV=300, and max_voltage starts at maxV.
        self.minV = 50
        self.maxV = 300
        self.max_voltage = self.maxV

    def connect_device(self, port_name):
        self.calls.append(("connect_device", port_name))
        return self.connected

    def get_device_status(self):
        self.calls.append(("get_device_status",))

    def set_max_voltage(self, v):
        self.calls.append(("set_max_voltage", v))
        # Mirrors the real PyCtrlDPP.set_max_voltage(): silently leaves
        # max_voltage unchanged when out of [minV, maxV] -- confirmed in
        # test_api_dpp_contract.py's test_set_max_voltage_* tests.
        if self.minV <= v <= self.maxV:
            self.max_voltage = v

    def load_precalibration(self, operation_mode="v"):
        self.calls.append(("load_precalibration", operation_mode))
        return True

    def load_infl_matrix(self, abs_path="", operation_mode="h"):
        self.calls.append(("load_infl_matrix", abs_path, operation_mode))
        return True

    def load_flat_field(self, abs_path="", operation_mode="v"):
        self.calls.append(("load_flat_field", abs_path, operation_mode))
        return True

    def apply_flat_field(self):
        self.calls.append(("apply_flat_field",))

    def apply_phases(self, phases):
        self.calls.append(("apply_phases", phases))

    def zero_outputs(self):
        self.calls.append(("zero_outputs",))

    def get_approx_ampls(self, get_all_approximated=False):
        self.calls.append(("get_approx_ampls", get_all_approximated))
        return list(self._approx_ampls)

    def set_pins_volts(self, pins2voltages):
        self.calls.append(("set_pins_volts", pins2voltages))
        applied = {}
        for pin, voltage in pins2voltages.items():
            if isinstance(pin, int) and isinstance(voltage, int) and 0 <= pin <= 63:
                applied[pin] = voltage
        return applied

    def close(self):
        self.calls.append(("close",))


@pytest.fixture
def patched_pyctrldpp(monkeypatch):
    """Swap the real PyCtrlDPP for FakePyCtrlDPP inside
    navigate.model.devices.APIs.phaseform.phaseplate.
    """
    import navigate.model.devices.APIs.phaseform.phaseplate as phaseform_dpp_module

    def fake_initialize(verbose_debug_info=False, print_license_note=True):
        return FakePyCtrlDPP(print_debug_info=verbose_debug_info)

    monkeypatch.setattr(phaseform_dpp_module.api_dpp, "initialize", fake_initialize)
    return phaseform_dpp_module


def make_plate(**overrides):
    """Build a PhaseformDPP with sane defaults, overridable per test."""

    kwargs = dict(port="COM4", calibration_file_path="/tmp/Calibration.json")
    kwargs.update(overrides)
    return PhaseformDPP(**kwargs)


# Module-level mode_orders / mode_names tables
# ---------------------------------------------------------------------------


def test_mode_orders_and_mode_names_have_matching_lengths():
    assert len(mode_orders) == len(mode_names) == 36


def test_mode_orders_index_0_is_piston():
    assert mode_orders[0] == (0, 0)
    assert mode_names[0] == ""


def test_mode_orders_index_with_corresponding_names():
    assert mode_orders[1] == (-1, 1)
    assert mode_names[1] == "Vert. tilt"
    assert mode_orders[4] == (0, 2)
    assert mode_names[4] == "Defocus"


# __init__
# ---------------------------------------------------------------------------


def test_init_stores_constructor_args(patched_pyctrldpp):
    plate = make_plate(
        operation_mode="h", max_voltage=250, n_modes=10, load_default=True
    )

    assert plate.port == "COM4"
    assert plate.calibration_file_path == "/tmp/Calibration.json"
    assert plate.operation_mode == "h"
    assert plate.max_voltage == 250
    assert plate.n_modes == 10
    assert plate.load_default is True


def test_init_defaults_match_documented_values(patched_pyctrldpp):
    plate = make_plate()

    assert plate.operation_mode == "v"
    assert plate.max_voltage == 300
    assert plate.n_modes == 36
    assert plate.load_default is False


def test_init_connect_true_by_default_calls_connect(patched_pyctrldpp):
    plate = make_plate()

    assert plate._is_connected is True
    assert ("connect_device", "COM4") in plate.dpp.calls


def test_init_connect_false_does_not_connect(patched_pyctrldpp):
    plate = make_plate(connect=False)

    assert plate._is_connected is False
    assert plate.dpp.calls == []


@pytest.mark.parametrize("empty_value", ["", None])
def test_init_rejects_empty_calibration_path(patched_pyctrldpp, empty_value):
    with pytest.raises(ValueError, match="Calibration file path"):
        PhaseformDPP(port="COM4", calibration_file_path=empty_value)


@pytest.mark.parametrize("empty_value", ["", None])
def test_init_rejects_empty_port(patched_pyctrldpp, empty_value):
    with pytest.raises(ValueError, match="Port"):
        PhaseformDPP(port=empty_value, calibration_file_path="/tmp/Calibration.json")


def test_init_forwards_print_debug_info_to_pyctrldpp(patched_pyctrldpp):
    plate = make_plate(print_debug_info=True)

    assert plate.dpp.print_debug_info is True


# connect() -- handshake + max voltage
# ---------------------------------------------------------------------------


def test_connect_calls_happen_in_the_right_order(patched_pyctrldpp):
    plate = make_plate()

    call_names = [c[0] for c in plate.dpp.calls]
    assert call_names == [
        "connect_device",
        "get_device_status",
        "set_max_voltage",
        "load_infl_matrix",
        "load_flat_field",
    ]


def test_connect_device_status_queried_after_handshake_succeeds(patched_pyctrldpp):
    plate = make_plate()

    assert plate.dpp.calls.index(("get_device_status",)) > plate.dpp.calls.index(
        ("connect_device", "COM4")
    )


def test_connect_raises_connection_error_when_connect_device_raises(
    patched_pyctrldpp, monkeypatch
):
    def raising_connect(self, port_name):
        raise OSError("port busy")

    monkeypatch.setattr(FakePyCtrlDPP, "connect_device", raising_connect)

    with pytest.raises(ConnectionError):
        make_plate()


def test_connect_raises_connection_error_when_connect_device_returns_false(
    patched_pyctrldpp, monkeypatch
):
    monkeypatch.setattr(FakePyCtrlDPP, "connect_device", lambda self, port_name: False)

    with pytest.raises(ConnectionError):
        make_plate()


def test_connect_skips_set_max_voltage_when_max_voltage_is_none(patched_pyctrldpp):
    plate = make_plate(max_voltage=None)

    assert not any(c[0] == "set_max_voltage" for c in plate.dpp.calls)


def test_connect_forwards_max_voltage_value(patched_pyctrldpp):
    plate = make_plate(max_voltage=180)

    assert ("set_max_voltage", 180) in plate.dpp.calls


def test_connect_raises_when_dpp_silently_rejects_out_of_range_max_voltage(
    patched_pyctrldpp,
):
    """CONFIRMED against the real library (test_api_dpp_contract.py): the
    real PyCtrlDPP.set_max_voltage() never raises for an out-of-range value
    -- it silently leaves max_voltage unchanged and returns None either way.
    connect() must detect that itself by checking the device's actual
    post-call state, since there is no exception to catch and no return
    value to check."""
    with pytest.raises(RuntimeError, match="rejected"):
        make_plate(max_voltage=1000)


def test_connect_accepts_max_voltage_at_the_device_boundary_values(
    patched_pyctrldpp,
):
    """50 and 300 are themselves valid (inclusive range) on the real
    device -- confirm connect() doesn't misfire a false rejection at
    exactly the boundary."""
    plate_low = make_plate(max_voltage=50)
    assert plate_low.dpp.max_voltage == 50

    plate_high = make_plate(max_voltage=300)
    assert plate_high.dpp.max_voltage == 300


def test_connect_raises_runtime_error_when_set_max_voltage_raises(
    patched_pyctrldpp, monkeypatch
):
    def raising(self, v):
        raise OSError("voltage rejected by device")

    monkeypatch.setattr(FakePyCtrlDPP, "set_max_voltage", raising)

    with pytest.raises(RuntimeError, match="max voltage"):
        make_plate()


def test_connect_closes_the_connection_when_calibration_loading_fails(
    patched_pyctrldpp, monkeypatch
):
    """If the serial handshake succeeds but a later step (calibration
    loading, in this case) fails, connect() must not leave the underlying
    connection open and _is_connected lying about it -- the port should be
    closed and _is_connected reset to False before the error propagates."""
    plate = make_plate(connect=False)
    monkeypatch.setattr(
        FakePyCtrlDPP,
        "load_infl_matrix",
        lambda self, abs_path="", operation_mode="h": False,
    )

    with pytest.raises(RuntimeError, match="load_infl_matrix"):
        plate.connect()

    assert plate._is_connected is False
    assert ("close",) in plate.dpp.calls


def test_connect_called_again_reruns_calibration_load_but_not_handshake(
    patched_pyctrldpp,
):
    """Documented behavior: connect() only performs the serial handshake once
    (guarded by _is_connected), but re-runs calibration loading every call."""
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.connect()

    call_names = [c[0] for c in plate.dpp.calls]
    assert "connect_device" not in call_names
    assert call_names.count("load_infl_matrix") == 1
    assert call_names.count("load_flat_field") == 1


def test_reconnect_after_disconnect_reruns_the_full_handshake(patched_pyctrldpp):
    """A disconnect() followed by another connect() call is a real, expected
    lifecycle (not just a symmetric pair of no-ops) -- confirm it actually
    re-does the serial handshake, not just the calibration reload that
    test_connect_called_again_reruns_calibration_load_but_not_handshake
    confirms for an ALREADY-connected instance."""
    plate = make_plate()
    plate.disconnect()
    plate.dpp.calls.clear()

    plate.connect()

    call_names = [c[0] for c in plate.dpp.calls]
    assert "connect_device" in call_names
    assert "get_device_status" in call_names
    assert plate._is_connected is True


# connect() -- load_default=True branch
# ---------------------------------------------------------------------------


def test_connect_load_default_calls_load_precalibration(patched_pyctrldpp):
    plate = make_plate(load_default=True)

    assert ("load_precalibration", "v") in plate.dpp.calls
    assert not any(c[0] == "load_infl_matrix" for c in plate.dpp.calls)


def test_connect_load_default_raises_when_precalibration_raises(
    patched_pyctrldpp, monkeypatch
):
    def raising(self, operation_mode="v"):
        raise OSError("no default calibration on this install")

    monkeypatch.setattr(FakePyCtrlDPP, "load_precalibration", raising)

    with pytest.raises(RuntimeError, match="precalibration"):
        make_plate(load_default=True)


def test_connect_load_default_raises_when_precalibration_returns_false(
    patched_pyctrldpp, monkeypatch
):
    monkeypatch.setattr(
        FakePyCtrlDPP, "load_precalibration", lambda self, operation_mode="v": False
    )

    with pytest.raises(RuntimeError, match="load_precalibration"):
        make_plate(load_default=True)


# connect() -- normal (load_default=False) calibration branch
# ---------------------------------------------------------------------------


def test_connect_loads_influence_matrix_with_given_path_and_mode(patched_pyctrldpp):
    plate = make_plate(calibration_file_path="/tmp/Cal.json", operation_mode="h")

    assert ("load_infl_matrix", "/tmp/Cal.json", "h") in plate.dpp.calls


def test_connect_raises_when_load_infl_matrix_raises(patched_pyctrldpp, monkeypatch):
    def raising(self, abs_path="", operation_mode="h"):
        raise ValueError("malformed json")

    monkeypatch.setattr(FakePyCtrlDPP, "load_infl_matrix", raising)

    with pytest.raises(RuntimeError, match="influence matrix"):
        make_plate()


def test_connect_raises_when_load_infl_matrix_returns_false(
    patched_pyctrldpp, monkeypatch
):
    monkeypatch.setattr(
        FakePyCtrlDPP,
        "load_infl_matrix",
        lambda self, abs_path="", operation_mode="h": False,
    )

    with pytest.raises(RuntimeError, match="load_infl_matrix"):
        make_plate()


def test_connect_skips_flat_field_load_when_corrections_already_loaded(
    patched_pyctrldpp, monkeypatch
):
    """Combined *.json calibrations auto-load flat offsets inside
    load_infl_matrix() itself (real PyCtrlDPP behavior) -- corrections_loaded
    is already True by the time PhaseformDPP would call load_flat_field()."""

    def infl_matrix_sets_corrections(self, abs_path="", operation_mode="h"):
        self.calls.append(("load_infl_matrix", abs_path, operation_mode))
        self.corrections_loaded = True
        return True

    monkeypatch.setattr(FakePyCtrlDPP, "load_infl_matrix", infl_matrix_sets_corrections)

    plate = make_plate()

    assert not any(c[0] == "load_flat_field" for c in plate.dpp.calls)


def test_connect_raises_when_load_flat_field_raises(patched_pyctrldpp, monkeypatch):
    def raising(self, abs_path="", operation_mode="v"):
        raise ValueError("malformed flat offsets")

    monkeypatch.setattr(FakePyCtrlDPP, "load_flat_field", raising)

    with pytest.raises(RuntimeError, match="flat field offsets"):
        make_plate()


def test_connect_raises_when_load_flat_field_returns_false(
    patched_pyctrldpp, monkeypatch
):
    monkeypatch.setattr(
        FakePyCtrlDPP,
        "load_flat_field",
        lambda self, abs_path="", operation_mode="v": False,
    )

    with pytest.raises(RuntimeError, match="load_flat_field"):
        make_plate()


# disconnect()
# ---------------------------------------------------------------------------


def test_disconnect_closes_and_clears_connected_flag(patched_pyctrldpp):
    plate = make_plate()

    plate.disconnect()

    assert ("close",) in plate.dpp.calls
    assert plate._is_connected is False


def test_disconnect_twice_is_idempotent(patched_pyctrldpp):
    plate = make_plate()

    plate.disconnect()
    n_close_calls_after_first = sum(1 for c in plate.dpp.calls if c[0] == "close")
    plate.disconnect()
    n_close_calls_after_second = sum(1 for c in plate.dpp.calls if c[0] == "close")

    assert n_close_calls_after_first == 1
    assert n_close_calls_after_second == 1


def test_disconnect_without_ever_connecting_does_not_call_close(patched_pyctrldpp):
    plate = make_plate(connect=False)

    plate.disconnect()

    assert plate.dpp.calls == []


# flat()
# ---------------------------------------------------------------------------


def test_flat_forwards_to_apply_flat_field(patched_pyctrldpp):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.flat()

    assert plate.dpp.calls == [("apply_flat_field",)]


# zero_flatness()
# ---------------------------------------------------------------------------


def test_zero_flatness_sets_all_zero_flat_reference_then_applies_it(
    patched_pyctrldpp,
):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.zero_flatness()

    assert plate.dpp.corrections_loaded is True
    assert plate.dpp.flatten_field_coefficients.shape == (N_TERMS,)
    assert np.all(plate.dpp.flatten_field_coefficients == 0.0)
    assert plate.dpp.calls[-1] == ("apply_flat_field",)


# set_flat()
# ---------------------------------------------------------------------------


def test_set_flat_with_correct_length_pos(patched_pyctrldpp):
    plate = make_plate()

    plate.set_flat(pos=[0.1] * N_TERMS)

    assert plate.dpp.corrections_loaded is True
    assert list(plate.dpp.flatten_field_coefficients) == [0.1] * N_TERMS


@pytest.mark.parametrize("as_type", [list, tuple, np.array])
def test_set_flat_accepts_list_tuple_or_ndarray(patched_pyctrldpp, as_type):
    plate = make_plate()

    plate.set_flat(pos=as_type([0.0] * N_TERMS))

    assert plate.dpp.corrections_loaded is True


def test_set_flat_rejects_wrong_length_pos(patched_pyctrldpp):
    plate = make_plate()

    with pytest.raises(ValueError, match="does not match"):
        plate.set_flat(pos=[0.0] * 5)


def test_set_flat_rejects_when_neither_pos_nor_pos_path_given(patched_pyctrldpp):
    plate = make_plate()

    with pytest.raises(ValueError, match="pos"):
        plate.set_flat()


def test_set_flat_with_pos_path_forwards_to_load_flat_field(patched_pyctrldpp):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.set_flat(pos_path="/tmp/flat_offsets.json")

    assert ("load_flat_field", "/tmp/flat_offsets.json", "v") in plate.dpp.calls


def test_set_flat_with_pos_path_raises_when_load_fails(patched_pyctrldpp, monkeypatch):
    # Build the plate FIRST (connect()'s own internal load_flat_field() call
    # must succeed normally) -- only make the LATER, explicit set_flat() call
    # fail, by patching the fake's bound method after construction.
    plate = make_plate()
    plate.dpp.load_flat_field = lambda abs_path="", operation_mode="v": False

    with pytest.raises(RuntimeError, match="flat field reference"):
        plate.set_flat(pos_path="/tmp/flat_offsets.json")


def test_set_flat_pos_path_takes_precedence_over_pos_when_both_given(
    patched_pyctrldpp,
):
    plate = make_plate()
    plate.dpp.calls.clear()
    plate.dpp.corrections_loaded = False

    plate.set_flat(pos=[0.0] * N_TERMS, pos_path="/tmp/flat_offsets.json")

    # pos_path branch returns early -- flatten_field_coefficients is never
    # touched directly by PhaseformDPP in that branch.
    assert ("load_flat_field", "/tmp/flat_offsets.json", "v") in plate.dpp.calls
    assert plate.dpp.flatten_field_coefficients is None


# ---------------------------------------------------------------------------
# move_absolute_zero()
# ---------------------------------------------------------------------------


def test_move_absolute_zero_forwards_to_zero_outputs(patched_pyctrldpp):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.move_absolute_zero()

    assert plate.dpp.calls == [("zero_outputs",)]


# ---------------------------------------------------------------------------
# display_modes()
# ---------------------------------------------------------------------------


def test_display_modes_with_int_keyed_dict_passes_through_unchanged(
    patched_pyctrldpp,
):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.display_modes({2: 0.5, 4: -0.25})

    assert plate.dpp.calls == [("apply_phases", {2: 0.5, 4: -0.25})]


def test_display_modes_with_mn_tuple_keyed_dict_passes_through_unchanged(
    patched_pyctrldpp,
):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.display_modes({(2, 2): 0.5, (-2, 2): -0.5})

    assert plate.dpp.calls == [("apply_phases", {(2, 2): 0.5, (-2, 2): -0.5})]


@pytest.mark.parametrize("as_type", [list, np.array])
def test_display_modes_with_correct_length_list_pads_leading_piston_zero(
    patched_pyctrldpp, as_type
):
    plate = make_plate(n_modes=4)  # expects exactly 3 real-mode values
    plate.dpp.calls.clear()

    plate.display_modes(as_type([0.1, 0.2, 0.3]))

    assert len(plate.dpp.calls) == 1
    call_name, sent = plate.dpp.calls[0]
    assert call_name == "apply_phases"
    assert list(sent) == [0.0, 0.1, 0.2, 0.3]


def test_display_modes_rejects_too_short_list(patched_pyctrldpp):
    plate = make_plate(n_modes=4)  # expects exactly 3 values

    with pytest.raises(ValueError, match="does not match"):
        plate.display_modes([0.1, 0.2])

    assert not any(c[0] == "apply_phases" for c in plate.dpp.calls)


def test_display_modes_rejects_too_long_list(patched_pyctrldpp):
    plate = make_plate(n_modes=4)

    with pytest.raises(ValueError, match="does not match"):
        plate.display_modes([0.1, 0.2, 0.3, 0.4])


def test_display_modes_rejects_empty_list_when_n_modes_greater_than_1(
    patched_pyctrldpp,
):
    plate = make_plate(n_modes=4)

    with pytest.raises(ValueError):
        plate.display_modes([])


def test_display_modes_edge_case_n_modes_1_accepts_empty_list(patched_pyctrldpp):
    """n_modes=1 means 0 real modes (piston only) -- the only valid input is
    an empty list, which should NOT raise."""
    plate = make_plate(n_modes=1)
    plate.dpp.calls.clear()

    plate.display_modes([])

    call_name, sent = plate.dpp.calls[0]
    assert call_name == "apply_phases"
    assert list(sent) == [0.0]


def test_display_modes_edge_case_n_modes_0_always_rejects_list_input(
    patched_pyctrldpp,
):
    """n_modes=0 means expected_len = -1, which no real array length can ever
    equal -- list/array input becomes permanently unusable. Documenting this
    as an explicit edge case rather than letting it surprise someone."""
    plate = make_plate(n_modes=0)

    with pytest.raises(ValueError):
        plate.display_modes([])


def test_display_modes_accepts_negative_and_large_amplitudes(patched_pyctrldpp):
    """No local numeric range validation exists (matches real apply_phases()
    -- range clamping happens later, in the voltage solve) -- values should
    pass through unmodified."""
    plate = make_plate(n_modes=3)
    plate.dpp.calls.clear()

    plate.display_modes([-999.0, 1e6])

    _, sent = plate.dpp.calls[0]
    assert list(sent) == [0.0, -999.0, 1e6]


# get_modal_coefs()
# ---------------------------------------------------------------------------


def test_get_modal_coefs_returns_equal_length_coefs_and_indices(patched_pyctrldpp):
    plate = make_plate(n_modes=6)

    coefs, coefs_idx = plate.get_modal_coefs()

    assert len(coefs) == len(coefs_idx)


def test_get_modal_coefs_excludes_piston_and_uses_1_based_osa_indices(
    patched_pyctrldpp,
):
    plate = make_plate(n_modes=6)
    # FakePyCtrlDPP._approx_ampls = [0.0, 0.1, 0.2, 0.3, 0.4, ...]

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert coefs_idx == [1, 2, 3, 4, 5]


def test_get_modal_coefs_length_matches_n_modes_minus_1(patched_pyctrldpp):
    for n_modes in (1, 2, 6, 36):
        plate = make_plate(n_modes=n_modes)
        coefs, coefs_idx = plate.get_modal_coefs()
        assert len(coefs) == n_modes - 1
        assert len(coefs_idx) == n_modes - 1


def test_get_modal_coefs_default_calls_get_approx_ampls_with_get_all_approximated_false(
    patched_pyctrldpp,
):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.get_modal_coefs()

    assert ("get_approx_ampls", False) in plate.dpp.calls


def test_get_modal_coefs_only_commanded_false_calls_get_approx_ampls_with_get_all_approximated_true(
    patched_pyctrldpp,
):
    plate = make_plate()
    plate.dpp.calls.clear()

    plate.get_modal_coefs(only_commanded=False)

    assert ("get_approx_ampls", True) in plate.dpp.calls


# ---------------------------------------------------------------------------
# get_modal_coefs(only_commanded=True) -- filtering logic against the shapes
# the real PyCtrlDPP.get_approx_ampls(get_all_approximated=False) actually
# returns (see tests/test_api_dpp_contract.py for those shapes confirmed
# against the real library). FakePyCtrlDPP's canned _approx_ampls has no
# zero entries, so it can't exercise this filtering on its own -- these
# tests monkeypatch get_approx_ampls directly per-case instead.
# ---------------------------------------------------------------------------


def test_get_modal_coefs_only_commanded_excludes_zero_entries_in_list_form(
    patched_pyctrldpp,
):
    plate = make_plate(n_modes=6)
    # OSA 0..5: only indices 1 and 4 were actually commanded (non-zero).
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: [
        0.0,
        0.3,
        0.0,
        0.0,
        0.5,
        0.0,
    ]

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs_idx == [1, 4]
    assert coefs == [0.3, 0.5]


def test_get_modal_coefs_only_commanded_list_form_includes_last_valid_index_when_shorter_than_n_modes(
    patched_pyctrldpp,
):
    """Boundary check for the `min(self.n_modes, len(approx))` guard: when
    get_approx_ampls() returns fewer entries than n_modes (as the real
    PyCtrlDPP does before a full-length apply_phases() call -- confirmed in
    test_api_dpp_contract.py), a nonzero value sitting at the LAST valid
    index of the short array must still be included, not silently dropped
    by an off-by-one in the truncation bound."""
    plate = make_plate(n_modes=6)
    # Only 4 entries (OSA 0..3) -- shorter than n_modes=6. Index 3 is the
    # last index this array can legally provide.
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: [0.0, 0.0, 0.0, 0.7]

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs_idx == [3]
    assert coefs == [0.7]


def test_get_modal_coefs_only_commanded_handles_dict_with_int_keys(
    patched_pyctrldpp,
):
    plate = make_plate(n_modes=6)
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: {1: 0.05, 4: 0.1}

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs_idx == [1, 4]
    assert coefs == [0.05, 0.1]


def test_get_modal_coefs_only_commanded_handles_dict_with_mn_tuple_keys(
    patched_pyctrldpp,
):
    plate = make_plate(n_modes=6)
    # (-1, 1) is OSA index 1 (vertical tilt).
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: {(-1, 1): 0.05}

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs_idx == [1]
    assert coefs == [0.05]


def test_get_modal_coefs_only_commanded_excludes_piston_key_from_dict(
    patched_pyctrldpp,
):
    plate = make_plate(n_modes=6)
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: {
        0: 0.05
    }  # piston-only command

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs == []
    assert coefs_idx == []


def test_get_modal_coefs_only_commanded_dict_includes_explicit_zero_value(
    patched_pyctrldpp,
):
    """Dict form is presence-based, not value-based (matches the real
    library's contract, confirmed in test_api_dpp_contract.py): a key
    present with value 0.0 is still 'commanded', unlike the list-form
    zero-exclusion case above."""
    plate = make_plate(n_modes=6)
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: {1: 0.0, 4: 0.1}

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs_idx == [1, 4]
    assert coefs == [0.0, 0.1]


def test_get_modal_coefs_only_commanded_sorts_dict_results_by_osa_index(
    patched_pyctrldpp,
):
    plate = make_plate(n_modes=6)
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: {
        4: 0.1,
        1: 0.05,
    }  # deliberately out of order

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs_idx == [1, 4]
    assert coefs == [0.05, 0.1]


def test_get_modal_coefs_only_commanded_before_any_apply_returns_empty(
    patched_pyctrldpp,
):
    """Regression guard: before any apply_phases() call, the real
    PyCtrlDPP.get_approx_ampls() returns [0.0] (length 1) regardless of the
    get_all_approximated flag (confirmed in test_api_dpp_contract.py) --
    must not IndexError when only_commanded=True indexes past that."""
    plate = make_plate(n_modes=6)
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: [0.0]

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs == []
    assert coefs_idx == []


def test_get_modal_coefs_only_commanded_false_before_any_apply_returns_empty(
    patched_pyctrldpp,
):
    """Mirrors test_get_modal_coefs_only_commanded_before_any_apply_returns_empty,
    but for the only_commanded=False branch -- no equivalent test exists for
    it today. Confirmed against the real library in test_api_dpp_contract.py:
    get_approx_ampls(get_all_approximated=True) also returns just [0.0]
    before any apply_phases() call, so this exercises the same short-list
    slicing safety (coefs[1:self.n_modes]) that the only_commanded=True
    branch's min() guard exists for."""
    plate = make_plate(n_modes=6)
    plate.dpp.get_approx_ampls = lambda get_all_approximated=False: [0.0]

    coefs, coefs_idx = plate.get_modal_coefs(only_commanded=False)

    assert coefs == []
    assert coefs_idx == []


# get_wavefront_pix()
# ---------------------------------------------------------------------------


def test_get_wavefront_pix_returns_none(patched_pyctrldpp):
    plate = make_plate()

    assert plate.get_wavefront_pix() is None


def test_get_wavefront_fix_typo_name_no_longer_exists(patched_pyctrldpp):
    """Regression guard: this method was previously misnamed
    get_wavefront_fix (typo for get_wavefront_pix), which would break any
    caller reaching for the real IMOP_Mirror-shaped method name."""
    plate = make_plate()

    assert not hasattr(plate, "get_wavefront_fix")


# save_json() / load_json()
# ---------------------------------------------------------------------------


def test_save_json_writes_coefs_and_voltages(patched_pyctrldpp, tmp_path):
    plate = make_plate(n_modes=6)
    plate.dpp.voltages = np.arange(N_RAW_VOLTAGE_SLOTS, dtype="uint16")
    json_path = tmp_path / "correction.json"

    plate.save_json(path=str(json_path))

    with open(json_path) as f:
        data = json.load(f)

    assert set(data.keys()) == {"coefs", "voltages"}
    assert len(data["coefs"]) == 5  # n_modes - 1
    assert data["coefs"] == {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5}
    assert data["voltages"] == list(range(N_RAW_VOLTAGE_SLOTS))


def test_save_json_saves_every_real_coefficient_no_data_loss(patched_pyctrldpp):
    """Regression guard for the coefs/coefs_idx length-mismatch bug that
    silently dropped the last real mode's value on every save."""
    plate = make_plate(n_modes=36)

    coefs, coefs_idx = plate.get_modal_coefs()
    assert len(coefs) == len(coefs_idx) == 35


def test_save_json_with_name_resolves_path_and_creates_directory(
    patched_pyctrldpp, tmp_path
):
    plate = make_plate(calibration_file_path=str(tmp_path / "Calibration.json"))

    plate.save_json(name="my_correction")

    expected_path = tmp_path / "PhasePlate_files" / "my_correction.json"
    assert expected_path.is_file()


def test_save_json_without_path_or_name_prints_and_returns_none(
    patched_pyctrldpp, capsys
):
    plate = make_plate()

    result = plate.save_json()

    assert result is None
    assert "provide either" in capsys.readouterr().out.lower()


def test_load_json_round_trips_with_save_json(patched_pyctrldpp, tmp_path):
    plate = make_plate(n_modes=6)
    plate.dpp.voltages = np.arange(N_RAW_VOLTAGE_SLOTS, dtype="uint16")
    json_path = tmp_path / "correction.json"
    plate.save_json(path=str(json_path))
    plate.dpp.calls.clear()

    loaded = plate.load_json(path=str(json_path))

    assert loaded["coefs"] == {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5}
    call_name, sent_coefs = plate.dpp.calls[-1]
    assert call_name == "apply_phases"
    assert sent_coefs == {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}


def test_load_json_with_name_resolves_same_path_as_save_json(
    patched_pyctrldpp, tmp_path
):
    plate = make_plate(calibration_file_path=str(tmp_path / "Calibration.json"))
    plate.save_json(name="round_trip")

    loaded = plate.load_json(name="round_trip")

    assert "coefs" in loaded and "voltages" in loaded


def test_load_json_without_path_or_name_prints_and_returns_none(
    patched_pyctrldpp, capsys
):
    plate = make_plate()

    result = plate.load_json()

    assert result is None
    assert "provide either" in capsys.readouterr().out.lower()


def test_load_json_raises_file_not_found_error_with_clear_message(
    patched_pyctrldpp,
):
    plate = make_plate()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        plate.load_json(path="/tmp/definitely_does_not_exist_xyz123.json")


def test_load_json_raises_value_error_for_invalid_json_syntax(
    patched_pyctrldpp, tmp_path
):
    """A file that exists but isn't valid JSON at all (truncated write,
    hand-edited and broken, etc.) should fail with a clear ValueError naming
    the file -- not a raw json.JSONDecodeError with no file-path context."""
    json_path = tmp_path / "broken.json"
    json_path.write_text("{not valid json")

    plate = make_plate()

    with pytest.raises(ValueError, match="not valid JSON"):
        plate.load_json(path=str(json_path))


def test_load_json_raises_value_error_for_missing_coefs_key(
    patched_pyctrldpp, tmp_path
):
    """A JSON file that parses fine but doesn't have the shape load_json()
    expects (e.g. saved by something else, or from an older format) should
    fail with a clear ValueError naming the missing key -- not a raw,
    unwrapped KeyError."""
    json_path = tmp_path / "malformed.json"
    with open(json_path, "w") as f:
        json.dump({"voltages": [0] * 64}, f)  # "coefs" missing

    plate = make_plate()

    with pytest.raises(ValueError, match="coefs"):
        plate.load_json(path=str(json_path))


def test_load_json_raises_value_error_for_missing_voltages_key_with_use_saved_voltages(
    patched_pyctrldpp, tmp_path
):
    json_path = tmp_path / "malformed.json"
    with open(json_path, "w") as f:
        json.dump({"coefs": {"1": 0.05}}, f)  # "voltages" missing

    plate = make_plate()

    with pytest.raises(ValueError, match="voltages"):
        plate.load_json(path=str(json_path), use_saved_voltages=True)


def test_load_json_raises_value_error_when_top_level_json_is_not_a_dict(
    patched_pyctrldpp, tmp_path
):
    """Covers the TypeError branch: a syntactically valid JSON file whose
    top-level value isn't a dict at all (e.g. a bare list) can't be indexed
    by string key -- confirms that's wrapped the same way as a missing key,
    not left as a raw TypeError."""
    json_path = tmp_path / "not_a_dict.json"
    with open(json_path, "w") as f:
        json.dump([1, 2, 3], f)

    plate = make_plate()

    with pytest.raises(ValueError):
        plate.load_json(path=str(json_path))


def test_load_json_use_saved_voltages_calls_set_pins_volts_not_apply_phases(
    patched_pyctrldpp, tmp_path
):
    plate = make_plate()
    plate.dpp.voltages = np.arange(N_RAW_VOLTAGE_SLOTS, dtype="uint16")
    json_path = tmp_path / "correction.json"
    plate.save_json(path=str(json_path))
    plate.dpp.calls.clear()

    plate.load_json(path=str(json_path), use_saved_voltages=True)

    call_names = [c[0] for c in plate.dpp.calls]
    assert "set_pins_volts" in call_names
    assert "apply_phases" not in call_names


def test_load_json_use_saved_voltages_sends_all_64_channels_as_ints(
    patched_pyctrldpp, tmp_path
):
    plate = make_plate()
    plate.dpp.voltages = np.arange(N_RAW_VOLTAGE_SLOTS, dtype="uint16")
    json_path = tmp_path / "correction.json"
    plate.save_json(path=str(json_path))
    plate.dpp.calls.clear()

    plate.load_json(path=str(json_path), use_saved_voltages=True)

    _, sent = next(c for c in plate.dpp.calls if c[0] == "set_pins_volts")
    assert len(sent) == N_RAW_VOLTAGE_SLOTS
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in sent.items())


def test_load_json_use_saved_voltages_shifts_pin_numbers_for_a_63_length_save(
    patched_pyctrldpp, tmp_path
):
    """A save taken with a 63-length voltages array (N_CHANNELS -- matching
    what the real PyCtrlDPP.apply_phases() actually produces: one entry per
    real actuator channel, at array position i for hardware channel i+1)
    must have its positions shifted by +1 before being replayed through
    set_pins_volts(), whose dict keys ARE the real 1..63 channel numbers
    directly, with key 0 always reserved. CONFIRMED against real hardware
    and dpp_ctrl's digitize_voltages(..., ignored_indices=[0]) source --
    see test_hardware_dpp.py's test_save_and_load_json_round_trip_via_
    use_saved_voltages. The 64-length case
    (test_load_json_use_saved_voltages_sends_all_64_channels_as_ints above,
    already in set_pins_volts()'s own 0..63 slot numbering) must NOT be
    shifted -- these two tests are a matched pair guarding both branches of
    dpp.py's length-dependent pin_offset."""
    plate = make_plate()
    plate.dpp.voltages = np.arange(N_CHANNELS, dtype="uint16")
    json_path = tmp_path / "correction.json"
    plate.save_json(path=str(json_path))
    plate.dpp.calls.clear()

    plate.load_json(path=str(json_path), use_saved_voltages=True)

    _, sent = next(c for c in plate.dpp.calls if c[0] == "set_pins_volts")
    assert set(sent.keys()) == set(range(1, N_CHANNELS + 1))
    assert sent[1] == 0  # saved position 0 (channel 1) -> pin key 1
    assert sent[N_CHANNELS] == N_CHANNELS - 1  # saved position 62 -> pin key 63


def test_load_json_returns_full_data_regardless_of_branch(patched_pyctrldpp, tmp_path):
    plate = make_plate()
    json_path = tmp_path / "correction.json"
    plate.save_json(path=str(json_path))

    loaded_coefs_path = plate.load_json(path=str(json_path))
    loaded_voltages_path = plate.load_json(path=str(json_path), use_saved_voltages=True)

    assert set(loaded_coefs_path.keys()) == {"coefs", "voltages"}
    assert set(loaded_voltages_path.keys()) == {"coefs", "voltages"}


def test_save_json_path_takes_precedence_over_name_when_both_given(
    patched_pyctrldpp, tmp_path
):
    plate = make_plate(calibration_file_path=str(tmp_path / "Calibration.json"))

    explicit_path = tmp_path / "explicit.json"

    plate.save_json(path=str(explicit_path), name="should_be_ignored")

    assert explicit_path.is_file()
    assert not (tmp_path / "PhasePlate_files").exists()


def test_load_json_path_takes_precedence_over_name_when_both_given(
    patched_pyctrldpp, tmp_path
):
    plate = make_plate(calibration_file_path=str(tmp_path / "Calibration.json"))
    explicit_path = tmp_path / "explicit.json"
    plate.save_json(path=str(explicit_path))

    # "wrong-name" resolves to a file that doesn't exist -- if name ever won
    # instead of path here, this would raise FileNotFoundError.
    loaded = plate.load_json(path=str(explicit_path), name="wrong-name")

    assert "coefs" in loaded
