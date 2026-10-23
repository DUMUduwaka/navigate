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

"""Tests for TonyWilson: sweep scale (mirror amplitude limits), the frame plan,
the check frame after every fitted update, and the signal/data hand-shake."""

import logging
import queue
import threading
import time
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


def make_tony_wilson(
    monkeypatch, controller, armed, amplitude=0.2, start=None, iterations=1, steps=5
):
    monkeypatch.setattr(
        adaptive_optics,
        "ImageWriter",
        lambda *args, **kwargs: types.SimpleNamespace(save_image=lambda *a, **k: None),
    )
    names = [f"mode{i}" for i in range(32)]
    start = np.zeros(32) if start is None else np.asarray(start)
    settings = ListKeysDict(
        iterations=iterations,
        steps=steps,
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


GAUSS_WIDTH = 0.3


def peaked_metric(target, noise=0.0, seed=0):
    """Image metric peaking at `target` (a 32-value vector)."""
    rng = np.random.default_rng(seed)
    target = np.asarray(target, dtype=float)

    def metric(cmd):
        value = np.exp(-np.sum((np.asarray(cmd) - target) ** 2) / GAUSS_WIDTH)
        return float(value * (1 + noise * rng.normal()))

    return metric


def run_lockstep(tony_wilson, metric, deliver_late=False):
    """Run signal, image and data frame by frame, like the rig when the data
    thread keeps up. With deliver_late, each frame reaches the data thread one
    call after its queue entry was taken (common on real hardware).

    Returns the list of (schedule entry, mirror command) per signal frame."""
    model = tony_wilson.model
    controller = tony_wilson.mirror_controller
    tony_wilson.pre_func_data()
    frames = []
    k = 0
    while not tony_wilson.end_func_signal():
        model.frame_id = k
        entry = tony_wilson.schedule[
            min(tony_wilson.signal_id, len(tony_wilson.schedule) - 1)
        ]
        tony_wilson.in_func_signal()
        frames.append((entry, controller._last.copy()))
        model.data_buffer[k] = np.full((2, 2), metric(controller._last))
        if deliver_late:
            tony_wilson.in_func_data([])
        tony_wilson.in_func_data([k])
        done = tony_wilson.end_func_data()
        k += 1
        if done:
            break
    return frames


def sweep_commands(frames, coef):
    """Commands of the sweep frames of armed mode `coef`, per iteration."""
    out = {}
    for (kind, itr, c, _), cmd in frames:
        if kind == "sweep" and c == coef:
            out.setdefault(itr, []).append(cmd)
    return out


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

    frames = run_lockstep(tony_wilson, peaked_metric(np.zeros(32)))

    expected = np.linspace(-0.2, 0.2, 5).astype(np.float32)
    assert [cmd[3] for cmd in sweep_commands(frames, 0)[0]] == expected.tolist()


def test_sweep_scale_keeps_each_mode_inside_its_limits(monkeypatch):
    positive = [1.0] * 32
    negative = [-1.0] * 32
    positive[3], negative[3] = 0.05, -0.08  # narrow mode
    positive[7], negative[7] = 0.01, -0.01  # almost no range -> 0.9*0.01/0.2
    controller = LimitedController(positive, negative)
    start = np.zeros(32)
    start[4] = 0.9  # close to its +1.0 limit

    tony_wilson = make_tony_wilson(
        monkeypatch, controller, armed=[2, 3, 4, 7], amplitude=0.2, start=start
    )
    frames = run_lockstep(tony_wilson, peaked_metric(start))

    # 90% of the room left (SWEEP_LIMIT_MARGIN), floored at SWEEP_SCALE_MIN
    # (0.01); mode 7: 0.045 = 0.9 * 0.01 / 0.2
    np.testing.assert_allclose(
        tony_wilson.sweep_scale, [1.0, 0.225, 0.45, 0.045], rtol=1e-5
    )
    offsets = {
        c: [cmd[c] - start[c] for cmd in sweep_commands(frames, j)[0]]
        for j, c in enumerate((2, 3, 4))
    }
    assert max(abs(v) for v in offsets[2]) == pytest.approx(0.2)
    assert max(offsets[3]) <= 0.9 * 0.05 + 1e-6 and min(offsets[3]) >= -0.08 - 1e-6
    assert max(offsets[4]) <= 1.0 - 0.9 + 1e-6


def test_frame_plan_has_a_reference_and_a_check_after_every_sweep(monkeypatch):
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[3, 4], iterations=2, steps=3
    )

    kinds = [kind for kind, *_ in tony_wilson.schedule]

    assert tony_wilson.total_frame_num == tony_wilson.get_tw_frame_num() == 17
    assert kinds == ["reference"] + (["sweep"] * 3 + ["check"]) * 4


@pytest.mark.parametrize("iterations", [1, 2])
def test_every_armed_mode_is_updated_in_every_iteration(monkeypatch, iterations):
    armed = list(range(3, 11))  # 8 modes, as on the rig
    target = np.zeros(32)
    target[armed] = [0.08, -0.06, 0.05, -0.04, 0.07, -0.05, 0.06, -0.07]
    tony_wilson = make_tony_wilson(
        monkeypatch,
        EchoController(),
        armed=armed,
        amplitude=0.2,
        start=np.zeros(32),
        iterations=iterations,
        steps=3,
    )

    run_lockstep(tony_wilson, peaked_metric(target))

    assert tony_wilson.done_all
    assert len(tony_wilson.decisions) == iterations * len(armed)
    assert all(d["accepted"] for d in tony_wilson.decisions)
    for c in armed:  # every mode moved toward its optimum, none left at 0
        assert abs(tony_wilson.best_coefs[c] - target[c]) < abs(target[c])


def test_single_mode_single_iteration_is_updated(monkeypatch):
    target = np.zeros(32)
    target[4] = 0.05
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[4], start=np.zeros(32), steps=3
    )

    run_lockstep(tony_wilson, peaked_metric(target))

    assert tony_wilson.best_coefs[4] == pytest.approx(0.05, abs=0.01)


def test_bad_update_is_rejected_and_the_next_sweep_starts_from_the_old_value(
    monkeypatch, caplog
):
    target = np.zeros(32)
    target[3] = 0.05
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[3, 4], start=np.zeros(32), steps=3
    )
    original = tony_wilson.process_data

    def wrong_for_first_mode(coef, mode="poly"):
        update, r_2 = original(coef, mode)
        return (-0.2, 1.0) if coef == 0 else (update, r_2)

    tony_wilson.process_data = wrong_for_first_mode
    caplog.set_level(logging.INFO)

    frames = run_lockstep(tony_wilson, peaked_metric(target))

    first = tony_wilson.decisions[0]
    assert first["mode"] == "mode3" and not first["accepted"]
    assert tony_wilson.best_coefs[3] == 0.0
    assert all(cmd[3] == 0.0 for cmd in sweep_commands(frames, 1)[0])
    assert "rejected" in caplog.text


def test_every_sweep_is_taken_around_the_accepted_coefficients(monkeypatch):
    armed = [3, 4, 5]
    target = np.zeros(32)
    target[armed] = [0.05, -0.04, 0.03]
    tony_wilson = make_tony_wilson(
        monkeypatch,
        EchoController(),
        armed=armed,
        start=np.zeros(32),
        iterations=2,
        steps=3,
    )
    accepted_after = []
    original = tony_wilson.decide

    def spy(metric, itr, coef):
        original(metric, itr, coef)
        accepted_after.append(tony_wilson.best_coefs.copy())

    tony_wilson.decide = spy

    frames = run_lockstep(tony_wilson, peaked_metric(target))

    sweeps = [cmd for (kind, *_), cmd in frames if kind == "sweep"]
    bases = [np.zeros(32)] + accepted_after[:-1]
    for k, base in enumerate(bases):  # sweep k, its 3 frames
        swept = armed[k % len(armed)]
        for cmd in sweeps[3 * k : 3 * k + 3]:
            others = np.delete(np.arange(32), swept)
            np.testing.assert_array_equal(cmd[others], np.float32(base)[others])
        assert sweeps[3 * k + 1][swept] == np.float32(base[swept])  # middle step


def test_signal_thread_waits_for_the_data_thread(monkeypatch):
    """Real threads: frames reach the data thread late; the signal thread must
    wait for each fit and decision, so the result equals the lock-step run."""
    armed = [3, 4]
    target = np.zeros(32)
    target[armed] = [0.05, -0.04]
    reference = make_tony_wilson(
        monkeypatch, EchoController(), armed=armed, start=np.zeros(32), steps=3
    )
    run_lockstep(reference, peaked_metric(target))

    controller = EchoController()
    tony_wilson = make_tony_wilson(
        monkeypatch, controller, armed=armed, start=np.zeros(32), steps=3
    )
    tony_wilson.pre_func_data()
    model = tony_wilson.model
    metric = peaked_metric(target)
    camera = queue.Queue()

    def data_thread():
        while True:
            frame_id = camera.get()
            if frame_id is None:
                return
            time.sleep(0.02)  # slow camera read-out and processing
            tony_wilson.in_func_data([frame_id])
            if tony_wilson.end_func_data():
                return

    worker = threading.Thread(target=data_thread)
    worker.start()
    k = 0
    while not tony_wilson.end_func_signal():
        model.frame_id = k
        tony_wilson.in_func_signal()
        model.data_buffer[k] = np.full((2, 2), metric(controller._last))
        camera.put(k)
        k += 1
    camera.put(None)
    worker.join(timeout=10)

    assert not tony_wilson.aborted
    assert tony_wilson.done_all
    np.testing.assert_array_equal(tony_wilson.best_coefs, reference.best_coefs)


def test_wait_timeout_stops_the_run(monkeypatch, caplog):
    monkeypatch.setattr(adaptive_optics, "WAIT_TIMEOUT_S", 0.2)
    controller = EchoController()
    tony_wilson = make_tony_wilson(
        monkeypatch, controller, armed=[3], start=np.zeros(32), steps=3
    )
    caplog.set_level(logging.ERROR)

    for _ in range(10):  # no data thread: the check waits for a fit forever
        if tony_wilson.end_func_signal():
            break
        tony_wilson.in_func_signal()

    assert tony_wilson.aborted and tony_wilson.end_func_signal()
    assert tony_wilson.signal_id == 4  # reference + 3 sweep steps
    np.testing.assert_array_equal(controller.commands[-1], np.zeros(32))
    assert "stopping the run" in caplog.text


def test_stop_ends_a_wait_without_aborting(monkeypatch):
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[3], start=np.zeros(32), steps=3
    )
    for _ in range(4):  # reference + 3 sweep steps
        tony_wilson.in_func_signal()
    tony_wilson.model.stop_acquisition = True

    start = time.monotonic()
    assert tony_wilson.in_func_signal() is True
    assert time.monotonic() - start < 1.0
    assert not tony_wilson.aborted


def test_repeated_mirror_failures_stop_the_run(monkeypatch, caplog):
    class NoReadback(EchoController):
        def get_modal_coefs(self):
            return (np.zeros(32, dtype=np.float32), list(range(1, 33)))

    tony_wilson = make_tony_wilson(
        monkeypatch, NoReadback(), armed=[3], start=np.zeros(32), steps=3
    )
    caplog.set_level(logging.ERROR)

    for _ in range(10):
        if tony_wilson.end_func_signal():
            break
        tony_wilson.in_func_signal()

    assert tony_wilson.aborted
    assert tony_wilson.signal_id == 1  # only the all-zero reference frame passed
    assert "3 mirror updates in a row failed" in caplog.text


def test_frame_arriving_after_its_queue_entry_is_processed(monkeypatch):
    """develop raised UnboundLocalError ('coef') when a frame arrived one data
    call after its queue entry, which is common on real hardware."""
    target = np.zeros(32)
    target[[3, 4]] = [0.05, 0.0]
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[3, 4], start=np.zeros(32)
    )

    run_lockstep(tony_wilson, peaked_metric(target), deliver_late=True)

    assert tony_wilson.done_all
    assert tony_wilson.frames_done == tony_wilson.total_frame_num == 13
    assert tony_wilson.best_coefs[3] == pytest.approx(0.05, abs=0.01)


def test_accepted_shape_is_written_once_at_the_end(monkeypatch):
    controller = EchoController()
    tony_wilson = make_tony_wilson(monkeypatch, controller, armed=[3, 4])

    frames = run_lockstep(tony_wilson, peaked_metric(np.zeros(32)))

    assert len(controller.commands) == len(frames) + 1
    np.testing.assert_array_equal(controller.commands[-1], tony_wilson.best_coefs)


def test_stopped_run_applies_the_accepted_shape(monkeypatch):
    controller = EchoController()
    tony_wilson = make_tony_wilson(monkeypatch, controller, armed=[3])
    tony_wilson.pre_func_data()
    tony_wilson.best_coefs[3] = 0.04
    tony_wilson.model.stop_acquisition = True

    tony_wilson.end_func_data()
    tony_wilson.end_func_data()

    assert controller.commands[-1][3] == pytest.approx(0.04)
    assert len(controller.commands) == 1


def test_flat_metric_leaves_the_mode_unchanged_instead_of_nan(monkeypatch, caplog):
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[3, 4], start=np.zeros(32)
    )
    caplog.set_level(logging.WARNING)

    run_lockstep(tony_wilson, lambda cmd: 1.0)

    assert np.isfinite(tony_wilson.best_coefs).all()
    assert tony_wilson.best_coefs[3] == 0.0
    assert "did not change across the sweep" in caplog.text


def test_failed_fit_leaves_the_mode_unchanged(monkeypatch, caplog):
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[3, 4], start=np.zeros(32)
    )

    def no_convergence(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")

    monkeypatch.setattr(adaptive_optics, "curve_fit", no_convergence)
    caplog.set_level(logging.WARNING)

    run_lockstep(tony_wilson, peaked_metric(np.zeros(32)))

    assert tony_wilson.done_all
    assert tony_wilson.best_coefs[3] == 0.0
    assert "fit failed" in caplog.text


def test_trace_is_stored_under_the_swept_mode(monkeypatch):
    tony_wilson = make_tony_wilson(
        monkeypatch, EchoController(), armed=[3, 4], start=np.zeros(32)
    )

    run_lockstep(tony_wilson, peaked_metric(np.zeros(32)))

    assert list(tony_wilson.trace_list) == ["mode3", "mode4"]
