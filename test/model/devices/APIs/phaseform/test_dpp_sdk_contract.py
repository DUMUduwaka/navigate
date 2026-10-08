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

"""
Contract tests against the dpp_ctrl library (api_dpp + zernike_pol_calc).
"""

import inspect
import os

import numpy as np
import pytest

pytest.importorskip(
    "dpp_ctrl",
    reason="dpp_ctrl SDK not installed.",
)

from dpp_ctrl.api_dpp import PyCtrlDPP  # noqa: E402
from dpp_ctrl import zernike_pol_calc  # noqa: E402

from navigate.model.devices.APIs.phaseform.phaseplate import (  # noqa: E402
    mode_names,
    mode_orders,
)

calibration_file_path = os.path.join(
    os.path.dirname(inspect.getfile(PyCtrlDPP)),
    "calibrations",
    "DUMMY_Calibration.json",
)

number_of_modes = 91
number_of_channels = 63


def create_dpp_sdk():
    return PyCtrlDPP(print_debug_info=False)


def configured_dpp(operation_mode="v"):
    dpp = create_dpp_sdk()
    dpp.load_infl_matrix(abs_path=calibration_file_path, operation_mode=operation_mode)
    dpp.set_max_voltage(300)
    return dpp


@pytest.mark.parametrize(
    "method_name, required_params",
    [
        ("connect_device", ["port_name"]),
        ("load_infl_matrix", ["abs_path", "operation_mode"]),
        ("load_flat_field", ["abs_path", "operation_mode"]),
        ("load_precalibration", ["operation_mode"]),
        ("apply_phases", ["phases", "calculate_only"]),
        ("get_approx_ampls", ["get_all_approximated"]),
        ("zero_outputs", []),
        ("apply_flat_field", []),
        ("set_max_voltage", ["maximum_voltage"]),
        ("set_pins_volts", ["pins2voltages"]),
        ("close", []),
        ("get_device_status", []),
    ],
)
def test_pyctrldpp_method_accepts_expected_keyword_parameters(
    method_name, required_params
):
    method = getattr(PyCtrlDPP, method_name)
    sig = inspect.signature(method)
    actual_params = list(sig.parameters.keys())

    for expected in required_params:
        assert expected in actual_params, (
            f"PyCtrlDPP.{method_name} no longer has a '{expected}' parameter "
            f"-- actual signature: {sig}"
        )


def test_pyctrldpp_init_requires_explicit_print_debug_info():
    """PyCtrlDPP requires print_debug_info to be passed explicitly."""
    signature = inspect.signature(PyCtrlDPP.__init__)
    parameter = signature.parameters["print_debug_info"]

    assert parameter.default is inspect.Parameter.empty


def test_api_dpp_initialize_accepts_verbose_debug_info_and_returns_pyctrldpp():
    """Verify api_dpp.initialize accepts the expected keyword and returns PyCtrlDPP."""
    from dpp_ctrl import api_dpp

    signature = inspect.signature(api_dpp.initialize)
    assert "verbose_debug_info" in signature.parameters

    instance = api_dpp.initialize(verbose_debug_info=False, print_license_note=False)
    assert isinstance(instance, PyCtrlDPP)


def test_set_max_voltage_accepts_valid_value():
    dpp = create_dpp_sdk()
    dpp.set_max_voltage(200)
    assert dpp.max_voltage == 200


def test_set_max_voltage_rejects_value_above_maximum():
    """Verify an out-of-range value is rejected, not clamped to the boundary."""
    dpp = create_dpp_sdk()
    dpp.set_max_voltage(150)  # valid, non-boundary value first

    dpp.set_max_voltage(500)

    assert dpp.max_voltage == 150  # unchanged, NOT clamped to 300


def test_set_max_voltage_accepts_inclusive_boundary_values():
    """Verify the minimum and maximum supported voltages are accepted"""
    dpp = create_dpp_sdk()

    dpp.set_max_voltage(dpp.minV)
    assert dpp.max_voltage == dpp.minV

    dpp.set_max_voltage(dpp.maxV)
    assert dpp.max_voltage == dpp.maxV


def test_set_max_voltage_silently_rejects_out_of_range_values():
    """
    Verify the SDK silently rejects out of range volatages without
    changing max_voltage.
    """
    dpp = create_dpp_sdk()
    starting_max_voltage = dpp.max_voltage

    result_above = dpp.set_max_voltage(dpp.maxV + 1)
    result_below = dpp.set_max_voltage(dpp.minV - 1)

    assert result_above is None
    assert result_below is None
    assert dpp.max_voltage == starting_max_voltage


def test_pyctrldpp_initializes_with_expected_default_values():
    """Verify a new PyCtrlDPP instance starts with the expected values."""
    dpp = create_dpp_sdk()

    assert dpp.influence_matrix is None
    assert dpp.corrections_loaded is False
    assert dpp.default_calibrations_path


def test_fresh_instance_max_osa_index_default_is_35():
    dpp = create_dpp_sdk()
    assert dpp.max_osa_index == 35


def test_load_infl_matrix_on_real_json_returns_true_and_correct_shape():
    dpp = create_dpp_sdk()

    loaded = dpp.load_infl_matrix(abs_path=calibration_file_path, operation_mode="v")

    assert loaded is True
    assert isinstance(dpp.influence_matrix, np.ndarray)
    assert dpp.influence_matrix.shape == (number_of_modes, number_of_channels)
    assert dpp.influence_matrix.dtype == np.float64


def test_load_infl_matrix_from_json_loads_flat_field_corrections():
    """
    Verify load_infl_matrix() auto loads flat field corrections
    from json calibrations.
    """
    dpp = create_dpp_sdk()

    assert dpp.corrections_loaded is False

    dpp.load_infl_matrix(abs_path=calibration_file_path, operation_mode="v")

    assert dpp.corrections_loaded is True
    assert dpp.flatten_field_coefficients.shape == (number_of_modes,)


def test_max_osa_index_does_not_update_after_loading_calibration():
    dpp = configured_dpp()

    assert dpp.influence_matrix.shape[0] - 1 == 90
    assert dpp.max_osa_index == 35


def test_piston_row_of_loaded_influence_matrix_is_all_zero():
    dpp = configured_dpp()

    assert np.all(dpp.influence_matrix[0] == 0.0)


def _solve(dpp, target):
    dpp.apply_phases(list(target), calculate_only=True)
    return dpp.corrected_voltages.copy(), dpp.get_approx_ampls(
        get_all_approximated=True
    )


def test_calculate_only_computes_voltages_without_hardware():
    dpp = configured_dpp()
    target = [0.0] * number_of_modes
    target[1] = 0.05  # OSA index 1 = vertical tilt

    voltages, approx = _solve(dpp, target)

    assert dpp.volts_calculated is True
    assert voltages.shape == (number_of_channels,)
    assert approx[0] == 0.0
    assert abs(approx[1] - 0.05) < 0.01


@pytest.mark.parametrize("input_index", [0, 1, 31, 62])
def test_digitize_voltages_maps_solved_array_position_i_to_hardware_channel_i_plus_1(
    input_index,
):
    """Verify input position i maps to hardware channel i + 1."""
    from dpp_ctrl.ampcom.dpp_comm import digitize_voltages

    physical_volts = np.zeros(number_of_channels, dtype=np.float64)
    physical_volts[input_index] = 100.0

    digitized = digitize_voltages(physical_volts, max_voltage=300)
    nonzero_indices = np.flatnonzero(digitized)

    assert digitized.shape == (64,)
    assert digitized[0] == 0
    assert np.array_equal(nonzero_indices, [input_index + 1])


@pytest.mark.parametrize("piston_value", [0.0, 0.001, 0.1, 0.5, 1.0, -1.0])
def test_piston_values_do_not_change_the_solution(piston_value):
    """Verify piston targets do not change the non-piston solver solution."""

    dpp = configured_dpp()

    baseline_target = [0.0] * number_of_modes
    baseline_target[1] = 0.05
    voltages_baseline, approx_baseline = _solve(dpp, baseline_target)

    piston_target = baseline_target.copy()
    piston_target[0] = piston_value
    voltages_piston, approx_piston = _solve(dpp, piston_target)

    assert np.array_equal(voltages_baseline, voltages_piston)
    assert np.array_equal(approx_baseline[1:], approx_piston[1:])


def test_get_approx_ampls_not_all_before_any_apply_phases_call():
    dpp = configured_dpp()
    assert dpp.get_approx_ampls(get_all_approximated=False) == [0.0]
    assert dpp.get_approx_ampls(get_all_approximated=True) == [0.0]


def test_get_approx_ampls_zeroes_uncommanded_modes_when_get_all_is_false():
    """Verify uncommanded modes are zeroed when get_all_approximated = False."""
    dpp = configured_dpp()

    target = [0.0] * number_of_modes
    target[1] = 0.05
    target[10] = 0.1

    dpp.apply_phases(target, calculate_only=True)

    approx_amplitudes = dpp.get_approx_ampls(get_all_approximated=False)

    assert isinstance(approx_amplitudes, list)
    assert len(approx_amplitudes) == number_of_modes

    nonzero_indices = [
        i for i, amplitude in enumerate(approx_amplitudes) if amplitude != 0
    ]

    assert nonzero_indices == [1, 10]

    assert approx_amplitudes[1] == pytest.approx(0.05, abs=0.01)
    assert approx_amplitudes[10] == pytest.approx(0.1, abs=0.02)


def _create_configured_plate():
    from navigate.model.devices.APIs.phaseform.phaseplate import PhaseformDPP

    plate = PhaseformDPP(
        port="COM_UNUSED",
        calibration_file_path=calibration_file_path,
        operation_mode="v",
        connect=False,
    )

    plate.dpp.load_infl_matrix(
        abs_path=calibration_file_path,
        operation_mode="v",
    )

    plate.dpp.set_max_voltage(300)

    return plate


def test_get_modal_coefs_returns_only_commanded_modes_by_default():
    """Verify get_modal_coefs() returns only modes commanded through the real SDK."""

    plate = _create_configured_plate()

    plate.dpp.apply_phases(
        {1: 0.05, 10: 0.1},
        calculate_only=True,
    )

    coefs, coefs_idx = plate.get_modal_coefs()

    assert coefs_idx == [1, 10]
    assert len(coefs) == 2

    assert coefs[0] == pytest.approx(0.05, abs=0.01)
    assert coefs[1] == pytest.approx(0.1, abs=0.02)


def test_get_modal_coefs_returns_all_modes_when_only_commanded_is_false():
    """Verify only_commanded=False returns coefficients for every non-piston mode."""
    plate = _create_configured_plate()

    target = [0.0] * number_of_modes
    target[1] = 0.05
    plate.dpp.apply_phases(target, calculate_only=True)

    coefs, coefs_idx = plate.get_modal_coefs(only_commanded=False)

    assert len(coefs) == plate.n_modes - 1
    assert coefs_idx == list(range(1, plate.n_modes))


@pytest.mark.parametrize(
    "fn_name, expected_params",
    [
        ("get_orders_osa_index_correspondence", ["max_order"]),
        ("get_classical_polynomial_name", ["mode", "short_names"]),
        ("osa2orders", ["osa_index"]),
        ("get_osa_standard_index", ["m", "n"]),
    ],
)
def test_zernike_pol_calc_functions_accept_expected_keyword_parameters(
    fn_name, expected_params
):
    """Verify Zernike helpers accept the keyword arguments used by PhaseformDPP."""
    fn = getattr(zernike_pol_calc, fn_name)
    signature = inspect.signature(fn)

    for parameter_name in expected_params:
        assert parameter_name in signature.parameters, (
            f"zernike_pol_calc.{fn_name} no longer has a "
            f"'{parameter_name}' parameter; actual signature: {signature}"
        )


def test_mode_orders_matches_osa2orders_for_every_entry():
    for osa_index, order in enumerate(mode_orders):
        assert order == zernike_pol_calc.osa2orders(osa_index), (
            f"mode_orders[{osa_index}] = {order} does not match "
            f"osa2orders({osa_index}) = {zernike_pol_calc.osa2orders(osa_index)}"
        )


def test_mode_names_matches_get_classical_polynomial_name_for_every_entry():
    for osa_index, (order, name) in enumerate(zip(mode_orders, mode_names)):
        expected_name = zernike_pol_calc.get_classical_polynomial_name(
            order, short_names=True
        )
        assert name == expected_name, (
            f"mode_names[{osa_index}] = {name!r} does not match "
            f"get_classical_polynomial_name({order}, short_names=True) "
            f"= {expected_name!r}"
        )
