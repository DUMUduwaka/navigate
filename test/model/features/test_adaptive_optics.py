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


import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from navigate.model.features.adaptive_optics import TonyWilson
from test.model.dummy import DummyModel


def make_mirror():
    mirror = MagicMock(
        spec=["flat", "display_modes", "get_modal_coefs", "get_wavefront_pix"]
    )
    mirror.get_modal_coefs.return_value = ([0.0] * 32, list(range(1, 33)))
    mirror.get_wavefront_pix.return_value = None
    return mirror


def make_ao_settings(modes_armed, **overrides):
    settings = {
        "iterations": 1,
        "steps": 3,
        "amplitude": 1.0,
        "metric": "Pixel Average",
        "from": "flat",
        "fitfunc": "poly",
        "modes_armed": modes_armed,
    }
    settings.update(overrides)
    return settings


class TestTonyWilsonFrameHandling(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.model.logger = MagicMock()
        self.model.event_queue = MagicMock()
        self.model.active_microscope.mirror = make_mirror()

    def _make_tw(self, modes_armed, **settings_overrides):
        self.model.configuration["experiment"]["AdaptiveOpticsParameters"] = {
            "save_report": False,
            "TonyWilson": make_ao_settings(modes_armed, **settings_overrides),
        }
        tw = TonyWilson(self.model)
        tw.pre_func_signal()
        tw.pre_func_data()
        return tw

    def test_in_func_data_resumes_after_missing_frame_without_crashing(self):
        """
        A frame missing on the first call and arriving on the second
        must not raise UnboundLocalError, and must process correctly once
        it arrives.
        """
        tw = self._make_tw({"A": True, "B": False})
        # fid, frame_num, itr, coef, step
        tw.tw_frame_queue.put((7, 4, 0, 0, 1))
        self.model.data_buffer[7] = 3.0

        # Frame 7 hasn't arrived in this batch -- must not raise, and must
        # retain the popped tuple's metadata for the next call.
        tw.in_func_data(frame_ids=[99])

        self.assertEqual(tw.f_frame_id, 7)
        self.assertEqual(tw.f_coef, 0)
        self.assertEqual(tw.f_step, 1)
        self.assertEqual(tw.frames_done, 0)

        # Frame 7 arrives now -- must not raise.
        tw.in_func_data(frame_ids=[7])

        self.assertEqual(tw.frames_done, 1)
        self.assertEqual(tw.f_frame_id, -1)
        self.assertEqual(list(tw.plot_data), [3.0])

    def test_full_run_fits_every_armed_mode_once_including_last(self):
        """
        n_coefs=2, n_steps=3, n_iter=1 (6 frames): both armed modes must
        be fitted exactly once, each on its own last frame -- including the
        true last mode of the true last iteration, which the one-frame-late
        design never fitted at all.
        """
        tw = self._make_tw({"A": True, "B": False, "C": True})
        self.assertEqual(tw.change_coef, [0, 2])

        tuples = [
            (0, 5, 0, 0, 0),
            (1, 4, 0, 0, 1),
            (2, 3, 0, 0, 2),
            (3, 2, 0, 1, 0),
            (4, 1, 0, 1, 1),
            (5, 0, 0, 1, 2),
        ]
        # mode A: [3,5,4], mode C: [5,6,2]
        means = [3.0, 5.0, 4.0, 5.0, 6.0, 2.0]
        for t in tuples:
            tw.tw_frame_queue.put(t)
        for i, m in enumerate(means):
            self.model.data_buffer[i] = m

        for i in range(6):
            tw.in_func_data(frame_ids=[i])
            tw.end_func_data()

        self.assertTrue(tw.done_all)
        self.assertEqual(set(tw.trace_list.keys()), {"A", "C"})
        self.assertEqual(tw.best_coefs[1], 0.0)
        self.assertNotEqual(tw.best_coefs[0], 0.0)
        self.assertNotEqual(tw.best_coefs[2], 0.0)

        events = [c.args[0][0] for c in self.model.event_queue.put.call_args_list]
        self.assertIn("mirror_update", events)
        self.assertIn("tonywilson", events)

    def test_fit_failure_is_logged_and_skipped_not_fatal(self):
        """
        A mode whose fit raises must not propagate -- it should be
        logged and leave that mode's correction unchanged.
        """
        tw = self._make_tw({"A": True})

        tw.tw_frame_queue.put((0, 2, 0, 0, 0))
        tw.tw_frame_queue.put((1, 1, 0, 0, 1))
        tw.tw_frame_queue.put((2, 0, 0, 0, 2))
        for i, m in enumerate([1.0, 2.0, 1.0]):
            self.model.data_buffer[i] = m

        with patch(
            "navigate.model.features.adaptive_optics.curve_fit",
            side_effect=RuntimeError("Optimal parameters not found"),
        ):
            for i in range(3):
                tw.in_func_data(frame_ids=[i])

        self.assertEqual(tw.best_coefs[0], 0.0)  # fit failed, left unchanged
        self.model.logger.warning.assert_called()
        self.assertTrue(np.all(np.isnan(tw.trace_list["A"]["y_fit"])))

    def _mirror_update_payloads(self):
        return [
            c.args[0][1]
            for c in self.model.event_queue.put.call_args_list
            if c.args[0][0] == "mirror_update"
        ]

    def test_final_apply_readback_reports_achieved_coefs_distinct_from_commanded(
        self,
    ):
        """
        After the correction loop finishes, the mirror must be re-applied
        and read back. The reported "coefs" must be best_coefs_overall
        (what was actually sent to hardware), and "achieved_coefs" must be
        the real readback -- not silently identical to it.
        """
        tw = self._make_tw({"A": True})
        tw.mirror.get_modal_coefs.return_value = ([0.4242] * 32, list(range(1, 33)))

        tuples = [
            (0, 2, 0, 0, 0),
            (1, 1, 0, 0, 1),
            (2, 0, 0, 0, 2),
        ]
        means = [3.0, 6.0, 4.0]
        for t in tuples:
            tw.tw_frame_queue.put(t)
        for i, m in enumerate(means):
            self.model.data_buffer[i] = m

        for i in range(3):
            tw.in_func_data(frame_ids=[i])
            tw.end_func_data()

        self.assertTrue(tw.done_all)

        payloads = self._mirror_update_payloads()
        self.assertEqual(len(payloads), 1)
        coefs = list(payloads[0]["coefs"])
        achieved = payloads[0]["achieved_coefs"]

        self.assertEqual(achieved, [0.4242] * 32)
        self.assertEqual(coefs, list(tw.best_coefs_overall))
        self.assertNotEqual(coefs, achieved)

        call_names = [c[0] for c in tw.mirror.method_calls]
        self.assertLess(
            call_names.index("display_modes"),
            call_names.index("get_modal_coefs"),
        )

    def test_build_report_uses_best_coefs_overall_not_stale_best_coefs(self):
        """
        At an intermediate (non-final) iteration boundary, best_coefs can
        advance past best_coefs_overall if the just-fitted mode didn't beat
        the running best metric. The reported "coefs" must still match what
        was actually sent to hardware (best_coefs_overall), not the
        diverged best_coefs.
        """
        tw = self._make_tw({"A": True}, iterations=3)

        tuples = [
            (0, 8, 0, 0, 0),
            (1, 7, 0, 0, 1),
            (2, 6, 0, 0, 2),
            (3, 5, 1, 0, 0),
            (4, 4, 1, 0, 1),
            (5, 3, 1, 0, 2),
        ]
        # Iteration 0 peaks at 6.0 (new best, syncs best_coefs_overall).
        # Iteration 1 peaks at only 2.0 (does not beat it), so
        # best_coefs_overall freezes after iteration 0 while best_coefs
        # keeps accumulating through iteration 1.
        means = [3.0, 6.0, 4.0, 1.0, 2.0, 1.5]
        for t in tuples:
            tw.tw_frame_queue.put(t)
        for i, m in enumerate(means):
            self.model.data_buffer[i] = m

        for i in range(6):
            tw.in_func_data(frame_ids=[i])
            tw.end_func_data()

        self.assertFalse(tw.done_all)
        self.assertNotEqual(tw.best_coefs[0], tw.best_coefs_overall[0])

        payloads = self._mirror_update_payloads()
        self.assertEqual(len(payloads), 2)
        coefs = list(payloads[-1]["coefs"])

        self.assertEqual(coefs, list(tw.best_coefs_overall))
        self.assertNotEqual(coefs, list(tw.best_coefs))


if __name__ == "__main__":
    unittest.main()
