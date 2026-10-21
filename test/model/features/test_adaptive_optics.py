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

"""Tests for TonyWilson's per-mode sweep scale (mirror amplitude limits)."""

import logging
import queue
import types
from unittest import mock

import numpy as np
import pytest

from navigate.model.features import adaptive_optics


class ListKeysDict(dict):
    """Mimic multiprocessing DictProxy, whose keys() returns a list."""

    def keys(self):
        return list(super().keys())


class EchoController:
    """Mirror controller with IMOP_Mirror semantics: readback = last command."""

    n_modes = 32

    def __init__(self):
        self.commands = []
        self._last = np.zeros(32, dtype=np.float32)

    def flat(self):
        self._last = np.zeros(32, dtype=np.float32)

    def display_modes(self, coefs):
        self._last = np.asarray(coefs, dtype=np.float32).copy()
        self.commands.append(self._last.copy())

    def get_modal_coefs(self):
        return (self._last.copy(), list(range(1, 33)))

    def get_wavefront_pix(self):
        return None


class LimitedController(EchoController):
    def __init__(self, positive, negative):
        super().__init__()
        self.limits = (list(positive), list(negative))

    def get_amplitude_limits(self):
        return self.limits


def make_tony_wilson(monkeypatch, controller, armed, amplitude=0.2, start=None):
    monkeypatch.setattr(
        adaptive_optics,
        "ImageWriter",
        lambda *args, **kwargs: types.SimpleNamespace(save_image=lambda *a, **k: None),
    )
    names = [f"mode{i}" for i in range(32)]
    start = np.zeros(32) if start is None else np.asarray(start)
    settings = ListKeysDict(
        iterations=1,
        steps=5,
        amplitude=amplitude,
        metric="Pixel Max",
        fitfunc="poly",
        modes_armed=ListKeysDict({n: i in armed for i, n in enumerate(names)}),
        **{"from": "current"},
    )
    model = types.SimpleNamespace(
        configuration={
            "experiment": {
                "AdaptiveOpticsParameters": {"TonyWilson": settings},
                "MirrorParameters": {
                    "modes": ListKeysDict({n: float(v) for n, v in zip(names, start)})
                },
            }
        },
        active_microscope=types.SimpleNamespace(
            mirror=types.SimpleNamespace(mirror_controller=controller),
            current_exposure_time=0,
        ),
        logger=logging.getLogger("test"),
        event_queue=queue.Queue(),
        stop_acquisition=False,
        frame_id=0,
        data_buffer={},
    )
    tony_wilson = adaptive_optics.TonyWilson(model)
    tony_wilson.pre_func_signal()
    return tony_wilson


def run_signal(tony_wilson):
    while not tony_wilson.end_func_signal():
        tony_wilson.in_func_signal()


@pytest.mark.parametrize(
    "controller", [EchoController(), mock.MagicMock()], ids=["no_limits", "mock"]
)
def test_sweep_scale_is_one_without_amplitude_limits(monkeypatch, controller):
    if isinstance(controller, mock.MagicMock):
        controller.n_modes = 32
    tony_wilson = make_tony_wilson(monkeypatch, controller, armed=[2, 3, 7])

    assert tony_wilson.sweep_scale.dtype == np.float32
    assert (tony_wilson.sweep_scale == 1.0).all()


def test_sweep_matches_develop_formula_without_limits(monkeypatch):
    controller = EchoController()
    tony_wilson = make_tony_wilson(monkeypatch, controller, armed=[3], amplitude=0.2)

    run_signal(tony_wilson)

    expected = np.linspace(-0.2, 0.2, 5).astype(np.float32)
    assert [c[3] for c in controller.commands] == expected.tolist()


def test_sweep_scale_keeps_each_mode_inside_its_limits(monkeypatch):
    positive = [1.0] * 32
    negative = [-1.0] * 32
    positive[3], negative[3] = 0.05, -0.08  # narrow mode
    positive[7], negative[7] = 0.01, -0.01  # almost no range -> floor 0.1
    controller = LimitedController(positive, negative)
    start = np.zeros(32)
    start[4] = 0.9  # close to its +1.0 limit

    tony_wilson = make_tony_wilson(
        monkeypatch, controller, armed=[2, 3, 4, 7], amplitude=0.2, start=start
    )
    run_signal(tony_wilson)

    np.testing.assert_allclose(
        tony_wilson.sweep_scale, [1.0, 0.25, 0.5, 0.1], rtol=1e-5
    )
    swept = {
        c: [cmd[c] for cmd in controller.commands if cmd[c] != start[c]]
        for c in (2, 3, 4)
    }
    assert max(abs(v) for v in swept[2]) == pytest.approx(0.2)
    assert max(swept[3]) <= 0.05 + 1e-6 and min(swept[3]) >= -0.08 - 1e-6
    assert max(swept[4]) <= 1.0 + 1e-6
