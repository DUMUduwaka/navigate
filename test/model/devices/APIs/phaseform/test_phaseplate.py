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
Unit tests for PhaseformDPP (the navigate wrapper around the Phaseform SDK).

A fake ``dpp_ctrl`` package is installed in ``sys.modules`` before importing the
wrapper, so these tests run without the vendor SDK or hardware.
"""

import importlib
import json
import logging
import sys
import threading
import time
from types import ModuleType

import numpy as np
import pytest

PHASEPLATE = "navigate.model.devices.APIs.phaseform.phaseplate"


class FakeSDK:
    """Minimal stand-in for dpp_ctrl.api_dpp.PyCtrlDPP."""

    max_osa_index = 35
    minV = 50
    maxV = 300

    def __init__(self):
        self.calls = []
        self.max_voltage = self.maxV
        self.influence_matrix = np.zeros((36, 63))
        self.corrections_loaded = False
        self.approximated = [0.0] * 36
        self.ampls_limits_pos_list = []
        self.ampls_limits_neg_list = []
        self.voltages = np.zeros(63)
        self.fail = None
        self.limits_fail = None
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def _enter(self):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.0002)
        with self._guard:
            self.active -= 1

    def connect_device(self, port_name):
        self.calls.append(("connect_device", port_name))
        return True

    def get_device_status(self):
        pass

    def load_infl_matrix(self, abs_path, operation_mode):
        self.calls.append(("load_infl_matrix", abs_path))
        self.max_voltage = self.maxV  # the real SDK resets the max voltage here
        return True

    def load_flat_field(self, abs_path, operation_mode):
        self.calls.append(("load_flat_field", abs_path))
        self.corrections_loaded = True
        return True

    def set_max_voltage(self, maximum_voltage):
        self.calls.append(("set_max_voltage", maximum_voltage))
        if self.minV <= maximum_voltage <= self.maxV:
            self.max_voltage = maximum_voltage

    def apply_phases(self, phases, calculate_only=False, **kwargs):
        if self.fail is not None:
            raise self.fail
        self._enter()
        self.calls.append(("apply_phases", list(phases), calculate_only))
        padded = list(phases) + [0.0] * (36 - len(phases))
        self.approximated = [round(float(p) * 0.9, 4) for p in padded]

    def apply_flat_field(self):
        if self.fail is not None:
            raise self.fail
        self._enter()
        self.calls.append(("apply_flat_field",))
        self.approximated = [0.0] * 36

    def get_approx_ampls(self, get_all_approximated=False):
        return list(self.approximated)

    def get_ampls_limits(self):
        self.calls.append(("get_ampls_limits",))
        if self.limits_fail is not None:
            raise self.limits_fail
        self.ampls_limits_pos_list = [0.0] + [1.0 - 0.01 * i for i in range(1, 36)]
        self.ampls_limits_neg_list = [0.0] + [-0.9 + 0.01 * i for i in range(1, 36)]
        self.approximated = [0.5] * 36  # the real SDK leaves solver state behind
        return True

    def zero_outputs(self):
        self.calls.append(("zero_outputs",))

    def set_pins_volts(self, pins2voltages):
        self.calls.append(("set_pins_volts", pins2voltages))

    def close(self):
        self.calls.append(("close",))


@pytest.fixture(autouse=True)
def model_logger_propagates(monkeypatch):
    """Let caplog see this module's records even after navigate's logging.yml has
    been applied (it sets propagate=False on the "model" logger, e.g. whenever an
    earlier test in the same session builds a Model). Older pytest versions only
    capture through the root logger."""
    monkeypatch.setattr(logging.getLogger("model"), "propagate", True)


@pytest.fixture
def phaseplate(monkeypatch):
    dpp_ctrl = ModuleType("dpp_ctrl")
    api_dpp = ModuleType("dpp_ctrl.api_dpp")
    api_dpp.initialize = lambda verbose_debug_info=False: FakeSDK()
    zpc = ModuleType("dpp_ctrl.zernike_pol_calc")
    zpc.get_orders_osa_index_correspondence = lambda max_order=7: [
        (i, i) for i in range(36)
    ]
    zpc.get_classical_polynomial_name = lambda mode, short_names=False: str(mode)
    dpp_ctrl.api_dpp = api_dpp
    dpp_ctrl.zernike_pol_calc = zpc
    monkeypatch.setitem(sys.modules, "dpp_ctrl", dpp_ctrl)
    monkeypatch.setitem(sys.modules, "dpp_ctrl.api_dpp", api_dpp)
    monkeypatch.setitem(sys.modules, "dpp_ctrl.zernike_pol_calc", zpc)

    original = sys.modules.pop(PHASEPLATE, None)
    try:
        yield importlib.import_module(PHASEPLATE)
    finally:
        sys.modules.pop(PHASEPLATE, None)
        if original is not None:
            sys.modules[PHASEPLATE] = original


@pytest.fixture
def plate(phaseplate):
    return phaseplate.PhaseformDPP(
        port="COM_UNUSED", calibration_file_path="calibration.csv", connect=False
    )


def applied_phases(sdk):
    return [c[1] for c in sdk.calls if c[0] == "apply_phases"]


def test_n_modes_defaults_to_32_and_is_validated(phaseplate, plate):
    assert plate.n_modes == 32
    for bad in (0, 36):
        with pytest.raises(ValueError):
            phaseplate.PhaseformDPP(
                port="X", calibration_file_path="c.csv", connect=False, n_modes=bad
            )


def test_display_modes_sends_piston_slot_and_python_floats(plate):
    coefs = np.zeros(32, dtype=np.float32)
    coefs[3] = 0.25  # navigate slot 3 -> OSA index 4 (Defocus)

    plate.display_modes(coefs)

    (phases,) = applied_phases(plate.dpp)
    assert len(phases) == 33
    assert phases[0] == 0.0
    assert phases[4] == pytest.approx(0.25)
    assert all(type(p) is float for p in phases)


def test_get_modal_coefs_returns_exact_command_for_tonywilson_check(plate):
    applied = np.random.default_rng(0).normal(0, 0.1, 32).astype(np.float32)

    plate.display_modes(applied)
    coefs, indices = plate.get_modal_coefs()

    assert (applied == coefs).all()
    assert indices == list(range(1, 33))
    assert not (applied == plate.get_achieved_modal_coefs()).all()


def test_flat_resets_readback_to_zero(plate):
    plate.display_modes(np.full(32, 0.1, dtype=np.float32))
    plate.flat()

    assert ("apply_flat_field",) in plate.dpp.calls
    assert not plate.get_modal_coefs()[0].any()


def test_sdk_error_is_logged_not_raised_and_readback_kept(plate, caplog):
    first = np.full(32, 0.05, dtype=np.float32)
    plate.display_modes(first)
    plate.dpp.fail = RuntimeError("serial write failed")

    with caplog.at_level(logging.ERROR):
        plate.display_modes(np.full(32, 0.2, dtype=np.float32))
        plate.flat()

    assert "apply_phases failed" in caplog.text
    assert "apply_flat_field failed" in caplog.text
    assert (plate.get_modal_coefs()[0] == first).all()


def test_wrong_length_is_logged_and_not_sent(plate, caplog):
    with caplog.at_level(logging.ERROR):
        plate.display_modes([0.1] * 33)

    assert applied_phases(plate.dpp) == []
    assert "expected 32" in caplog.text


def test_dict_input_maps_osa_index_to_slot(plate, caplog):
    with caplog.at_level(logging.WARNING):
        plate.display_modes({4: 0.3, 40: 1.0})

    (phases,) = applied_phases(plate.dpp)
    assert phases[4] == pytest.approx(0.3)
    assert plate.get_modal_coefs()[0][3] == pytest.approx(0.3)
    assert "OSA index 40 ignored" in caplog.text


def test_lock_serializes_sdk_calls_from_two_threads(plate):
    def signal_thread():
        for i in range(100):
            plate.flat()
            plate.display_modes(np.full(32, i / 1000, dtype=np.float32))

    def data_thread():
        for _ in range(100):
            plate.display_modes(np.zeros(32, dtype=np.float32))

    threads = [
        threading.Thread(target=signal_thread),
        threading.Thread(target=data_thread),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert plate.dpp.max_active == 1


def test_save_and_load_json_round_trip(plate, tmp_path):
    command = np.random.default_rng(1).normal(0, 0.1, 32).astype(np.float32)
    plate.display_modes(command)
    path = str(tmp_path / "correction.json")

    plate.save_json(path=path)
    with open(path) as f:
        saved = json.load(f)
    assert list(saved["coefs"]) == [str(i) for i in range(1, 33)]
    assert np.float32(list(saved["coefs"].values())).tolist() == command.tolist()
    assert len(saved["achieved_coefs"]) == 32
    assert len(saved["voltages"]) == 63

    plate.flat()
    loaded = plate.load_json(path=path)

    assert np.float32(loaded).tolist() == command.tolist()
    assert (plate.get_modal_coefs()[0] == command).all()


def test_load_json_reads_older_files_and_skips_unknown_indices(plate, tmp_path, caplog):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"coefs": {"0": 0.9, "4": 0.2, "40": 1.0}, "voltages": []})
    )

    with caplog.at_level(logging.WARNING):
        loaded = plate.load_json(path=str(path))

    assert loaded[3] == pytest.approx(0.2)
    assert sum(1 for c in loaded if c) == 1
    assert "OSA index 0" in caplog.text and "OSA index 40" in caplog.text


def test_load_json_with_saved_voltages_sets_readback(plate, tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"coefs": {"4": 0.2}, "voltages": [10] * 63}))

    plate.load_json(path=str(path), use_saved_voltages=True)

    assert any(c[0] == "set_pins_volts" for c in plate.dpp.calls)
    assert plate.get_modal_coefs()[0][3] == pytest.approx(0.2)


def test_connect_sets_max_voltage_after_loading_calibration(phaseplate):
    plate = phaseplate.PhaseformDPP(
        port="COM4", calibration_file_path="calib.csv", max_voltage=150
    )

    names = [c[0] for c in plate.dpp.calls]
    assert names.index("load_infl_matrix") < names.index("set_max_voltage")
    assert plate.dpp.max_voltage == 150


def test_amplitude_limits_are_aligned_to_navigate_slots(phaseplate):
    plate = phaseplate.PhaseformDPP(port="COM4", calibration_file_path="calib.csv")

    positive, negative = plate.get_amplitude_limits()

    assert len(positive) == len(negative) == 32
    assert positive[0] == pytest.approx(0.99)  # OSA index 1
    assert negative[31] == pytest.approx(-0.58)  # OSA index 32


def test_limits_recompute_restores_last_command_in_calculate_only_mode(plate):
    plate.display_modes(np.full(32, 0.1, dtype=np.float32))

    plate._compute_amplitude_limits()

    last = [c for c in plate.dpp.calls if c[0] == "apply_phases"][-1]
    assert last[2] is True  # calculate_only
    assert last[1][1:] == pytest.approx([0.1] * 32)


def test_amplitude_limit_failure_does_not_fail_connect(phaseplate, monkeypatch, caplog):
    def broken_sdk(verbose_debug_info=False):
        sdk = FakeSDK()
        sdk.limits_fail = RuntimeError("solver failed")
        return sdk

    monkeypatch.setattr(sys.modules["dpp_ctrl.api_dpp"], "initialize", broken_sdk)

    with caplog.at_level(logging.WARNING):
        plate = phaseplate.PhaseformDPP(port="COM4", calibration_file_path="c.csv")

    assert plate.get_amplitude_limits() == (None, None)
    assert "could not compute amplitude limits" in caplog.text


def test_command_outside_limits_logs_warning(phaseplate, caplog):
    plate = phaseplate.PhaseformDPP(port="COM4", calibration_file_path="c.csv")
    coefs = np.zeros(32, dtype=np.float32)
    coefs[5] = 2.0

    with caplog.at_level(logging.WARNING):
        plate.display_modes(coefs)

    assert "outside the achievable range for slot(s) [5]" in caplog.text
    assert (plate.get_modal_coefs()[0] == coefs).all()  # the command is still sent
