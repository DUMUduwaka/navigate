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
Tests for PhaseformDPPMirror using a fake PhaseformDPP controller.

The Phaseform API module is replaced before importing the mirror wrapper so
the wrapper logic can be tested without the vendor SDK or physical hardware.
"""

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture
def mirror_configuration():
    return {
        "configuration": {
            "microscopes": {
                "scope-a": {
                    "mirror": {
                        "hardware": {
                            "port": "COM4",
                            "calibration_file_path": "fake-calibration.json",
                            "operation_mode": "v",
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def dpp_module(monkeypatch):
    fake_phaseform_api = ModuleType("navigate.model.devices.APIs.phaseform.phaseplate")

    class FakePhaseformDPP:
        instances = []

        def __init__(self, **kwargs):
            self.calls = []
            self.n_modes = None
            self.init_kwargs = kwargs
            self.__class__.instances.append(self)

        def flat(self):
            self.calls.append(("flat", {}))

        def move_absolute_zero(self):
            self.calls.append(("move_absolute_zero", {}))

        def set_flat(self, pos=None, pos_path=None):
            self.calls.append(("set_flat", {"pos": pos, "pos_path": pos_path}))

        def display_modes(self, coefs):
            self.calls.append(("display_modes", {"coefs": coefs}))

        def get_modal_coefs(self, only_commanded=True):
            self.calls.append(
                (
                    "get_modal_coefs",
                    {"only_commanded": only_commanded},
                )
            )
            return ([0.1, 0.2], [1, 2])

        def load_json(self, path=None, name=None):
            self.calls.append(("load_json", {"path": path, "name": name}))
            return {
                "coefs": {"1": "0.5", "2": "-0.3"},
                "voltages": [0] * 63,
            }

        def save_json(self, path=None, name=None):
            self.calls.append(("save_json", {"path": path, "name": name}))

    fake_phaseform_api.PhaseformDPP = FakePhaseformDPP

    monkeypatch.setitem(
        sys.modules,
        "navigate.model.devices.APIs.phaseform.phaseplate",
        fake_phaseform_api,
    )

    monkeypatch.delitem(
        sys.modules,
        "navigate.model.devices.mirror.dpp",
        raising=False,
    )

    module = importlib.import_module("navigate.model.devices.mirror.dpp")

    return module, FakePhaseformDPP


def test_constructor_sets_controller_n_modes_and_flattens(
    dpp_module, mirror_configuration
):
    """Verify construction configures 33 modes and flattens the controller."""
    module, fake_class = dpp_module
    controller = fake_class()

    mirror = module.PhaseformDPPMirror("scope-a", controller, mirror_configuration)

    assert mirror.mirror_controller is controller
    assert controller.n_modes == 33
    assert controller.calls == [("flat", {})]


def test_constructor_raises_for_unknown_microscope(
    dpp_module,
    mirror_configuration,
):
    module, _ = dpp_module

    with pytest.raises(NameError, match="missing-scope"):
        module.PhaseformDPPMirror(
            "missing-scope",
            None,
            mirror_configuration,
        )


def test_forwarding_methods_dispatch_to_controller(
    dpp_module,
    mirror_configuration,
):
    """Verify mirror methods forward to the expected controller methods."""
    module, fake_class = dpp_module
    controller = fake_class()

    mirror = module.PhaseformDPPMirror(
        "scope-a",
        controller,
        mirror_configuration,
    )

    controller.calls.clear()

    mirror.flat()
    mirror.zero_flatness()
    mirror.set_positions_flat([1, 2, 3])
    mirror.display_modes([0.0, 0.5, 1.0])

    assert controller.calls == [
        ("flat", {}),
        ("move_absolute_zero", {}),
        ("set_flat", {"pos": [1, 2, 3], "pos_path": None}),
        ("display_modes", {"coefs": [0.0, 0.5, 1.0]}),
    ]


def test_get_modal_coefs_passes_only_commanded_false(
    dpp_module,
    mirror_configuration,
):
    """Verify the mirror requests the full modal decomposition from the controller."""
    module, fake_class = dpp_module
    controller = fake_class()

    mirror = module.PhaseformDPPMirror(
        "scope-a",
        controller,
        mirror_configuration,
    )

    controller.calls.clear()

    modal_coefs = mirror.get_modal_coefs()

    assert modal_coefs == ([0.1, 0.2], [1, 2])
    assert controller.calls == [("get_modal_coefs", {"only_commanded": False})]


def test_json_file_methods_forward_arguments_and_convert_loaded_coefs(
    dpp_module,
    mirror_configuration,
):
    """Verify JSON methods forward arguments and normalize loaded coefficients."""
    module, fake_class = dpp_module
    controller = fake_class()

    mirror = module.PhaseformDPPMirror(
        "scope-a",
        controller,
        mirror_configuration,
    )

    controller.calls.clear()

    loaded_from_path = mirror.set_from_json_file(path="/tmp/load-path.json")
    loaded_from_name = mirror.set_from_json_file(name="correction-a")
    mirror.save_json_file(path="/tmp/save-path.json")
    mirror.save_json_file(name="correction-b")

    assert loaded_from_path == {1: 0.5, 2: -0.3}
    assert loaded_from_name == {1: 0.5, 2: -0.3}

    assert controller.calls == [
        ("load_json", {"path": "/tmp/load-path.json", "name": None}),
        ("load_json", {"path": None, "name": "correction-a"}),
        ("save_json", {"path": "/tmp/save-path.json", "name": None}),
        ("save_json", {"path": None, "name": "correction-b"}),
    ]


def test_json_file_methods_require_path_or_name(
    dpp_module,
    mirror_configuration,
):
    """Verify JSON methods reject calls without a path or name."""
    module, fake_class = dpp_module
    controller = fake_class()

    mirror = module.PhaseformDPPMirror(
        "scope-a",
        controller,
        mirror_configuration,
    )

    controller.calls.clear()

    with pytest.raises(
        ValueError,
        match="Either 'path' or 'name' must be provided",
    ):
        mirror.set_from_json_file()

    with pytest.raises(
        ValueError,
        match="Either 'path' or 'name' must be provided",
    ):
        mirror.save_json_file()

    assert controller.calls == []


def test_get_connect_params_returns_required_hardware_fields(dpp_module):
    module, _ = dpp_module

    assert module.PhaseformDPPMirror.get_connect_params() == [
        "port",
        "calibration_file_path",
        "operation_mode",
    ]


def test_connect_forwards_hardware_params_to_controller(dpp_module):
    """Verify connect() forwards hardware parameters to PhaseformDPP."""
    module, fake_class = dpp_module

    connection = module.PhaseformDPPMirror.connect(
        port="COM4",
        calibration_file_path="fake-calibration.json",
        operation_mode="v",
    )

    assert isinstance(connection, fake_class)
    assert connection.init_kwargs == {
        "port": "COM4",
        "calibration_file_path": "fake-calibration.json",
        "operation_mode": "v",
    }
