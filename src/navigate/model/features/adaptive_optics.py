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
from queue import Queue
from copy import deepcopy
import threading
import time

# Third Party Imports
import numpy as np
from scipy.optimize import curve_fit

# Local imports
from navigate.config.configuration_schema import SettingSpec
from navigate.model.features.base import FeatureBase
from navigate.model.features.common_features import PrepareNextChannel
import navigate.model.analysis.image_contrast as img_contrast
from navigate.model.features.image_writer import ImageWriter

#: float: fraction of a mode's amplitude limit TonyWilson sweeps up to (mirrors
#: that report limits only, e.g. Phaseform DPP); single-mode limits overstate
#: the headroom left when other modes are already corrected.
SWEEP_LIMIT_MARGIN = 0.9

#: float: smallest sweep scale, as a fraction of the amplitude.
SWEEP_SCALE_MIN = 0.01

#: float: a check frame may be up to this fraction below the reference and still
#: be accepted, so frame-to-frame noise does not reject small good corrections.
CHECK_TOLERANCE = 0.02

#: float: seconds the signal thread waits for the data thread (a fit or a check
#: decision) before it stops the run; below navigate's 10 s camera timeout.
WAIT_TIMEOUT_S = 5.0

#: int: failed mirror updates in a row after which the run stops.
MAX_MIRROR_FAILURES = 3


def poly2(x, a, b, c):
    """Second order polynomial function

    Parameters
    ----------
    x : float
        x value
    a : float
        a value
    b : float
        b value
    c : float
        c value

    Returns
    -------
    float
        y value
    """
    return a * x**2 + b * x + c


def gauss(x, a, b, c, d):
    """Gaussian function

    Parameters
    ----------
    x : float
        x value
    a : float
        a value
    b : float
        b value
    c : float
        c value
    d : float
        d value

    Returns
    -------
    float
        y value
    """
    y = (x - b) / c
    y = np.exp(-(y**2))
    return a * y + d


def r_squared(y, y_fit):
    """Calculate the R^2 value

    Parameters
    ----------
    y : list
        y values
    y_fit : list
        y_fit values

    Returns
    -------
    float
        R^2 value
    """
    y_bar = np.mean(y)
    SS_res = np.sum((y - y_fit) ** 2)
    SS_tot = np.sum((y - y_bar) ** 2)

    return 1 - SS_res / SS_tot


def fourier_annulus(im, radius_1=0, radius_2=64):
    """Calculate the mean of the fourier transform of an annulus

    Parameters
    ----------
    im : array
        Image array
    radius_1 : int, optional
        Inner radius of the annulus, by default 0
    radius_2 : int, optional
        Outer radius of the annulus, by default 64

    Returns
    -------
    float
        Mean of the fourier transform of the annulus
    array
        Fourier transform of the annulus
    """

    x_, y_ = np.meshgrid(range(im.shape[1]), range(im.shape[0]))

    x_ = x_.astype(float) - x_.mean()
    y_ = y_.astype(float) - y_.mean()

    mask = x_**2 + y_**2 > radius_1**2
    mask *= x_**2 + y_**2 <= radius_2**2

    IM = np.fft.fftshift(np.fft.fft2(im))
    IM_abs = np.abs(IM)

    IM_mask = IM_abs * mask

    return np.mean(IM_mask), IM_mask


class TonyWilson(FeatureBase):
    """Tony Wilson iterative AO routine"""

    parameter_schema = {
        "verbose": SettingSpec(
            bool,
            default=False,
            label="Verbose",
            help_text="Print detailed adaptive-optics progress while the routine runs.",
        ),
    }

    def __init__(self, model, verbose=False):
        """Initialize the Tony Wilson iterative AO routine

        Parameters
        ----------
        model : Model
            Model object
        """
        #: int: Number of modes
        self.n_modes = None

        #: bool: True if all iterations are done, False otherwise
        self.done_all = False

        #: bool: True if the current iteration is done, False otherwise
        self.done_itr = False

        #: list: detailed report to save as JSON after
        self.report = []

        #: bool: True if you want to print the full output string each step
        self.verbose = verbose

        self.start_time = 0

        #: navigate.model.Model: Model object
        self.model = model

        self.tw_settings = self.model.configuration["experiment"][
            "AdaptiveOpticsParameters"
        ]["TonyWilson"]

        #: navigate.model.devices.mirrors.mirror_imop.ImagineOpticsMirror: Mirror object
        self.mirror_controller = self.model.active_microscope.mirror.mirror_controller

        #: int: Number of modes
        self.n_modes = self.mirror_controller.n_modes

        #: bool: whether to save report at the end of run
        self.save_report = self.model.configuration["experiment"][
            "AdaptiveOpticsParameters"
        ].get("save_report", False)

        #: list: List of coefficients to change
        self.change_coef = []
        modes_armed_dict = self.tw_settings["modes_armed"]

        #: list: List of mode names
        self.mode_names = modes_armed_dict.keys()
        for i, k in enumerate(self.mode_names):
            if modes_armed_dict[k]:
                self.change_coef += [i]
        self.n_coefs = len(self.change_coef)

        self.start_from = self.tw_settings["from"]

        self.metric = self.tw_settings["metric"]

        self.fit_func = self.tw_settings["fitfunc"]

        # Queue
        self.tw_frame_queue = Queue()
        self.tw_data_queue = Queue()

        # Image Writer
        self.image_writer = ImageWriter(model, sub_dir="AO_Frames")

        # target channel
        self.target_channel = 1

        self.config_table = {
            "signal": {
                "init": self.pre_func_signal,
                "main": self.in_func_signal,
                "end": self.end_func_signal,
            },
            "data": {
                "init": self.pre_func_data,
                "main": self.in_func_data,
                "end": self.end_func_data,
            },
            "node": {"node_type": "multi-step", "device_related": True},
        }

    def run(self, *args):
        """Run the Tony Wilson iterative AO routine

        Parameters
        ----------
        args[0] : dict
            Current microscope state.
        args[1] : dict
            Autofocus parameters

        """
        frame_num = self.get_tw_frame_num()
        if frame_num < 1:
            return

        self.model.addon_feature = [
            [{"name": PrepareNextChannel}, {"name": TonyWilson}]
        ]

        self.model.configuration["experiment"]["MicroscopeState"][
            "image_mode"
        ] = "customized"

        self.model.run_command("acquire")

    def get_tw_frame_num(self):
        """Frames per run: one reference frame, then for every iteration and armed
        mode the sweep steps plus one check frame."""
        n_iter = self.tw_settings["iterations"]
        n_steps = self.tw_settings["steps"]
        if self.n_coefs < 1 or n_iter < 1 or n_steps < 1:
            return 0
        return 1 + n_iter * self.n_coefs * (n_steps + 1)

    # don't need this?
    def get_steps(self, ranges, step_size):
        """Calculate the number of steps and the position offset

        Parameters
        ----------
        ranges : int
            Range of the scan
        step_size : int
            Step size

        Returns
        -------
        int
            Number of steps
        int
            Position offset
        """
        steps = ranges // step_size + 1
        pos_offset = (steps // 2) * step_size + step_size
        return steps, pos_offset

    def build_schedule(self):
        """Return the frame plan: (kind, iteration, armed index, step) per frame.

        kind is "reference" (once, at the start), "sweep" (n_steps per mode) or
        "check" (once per mode, with that mode's fitted update applied).
        """
        schedule = [("reference", None, None, None)]
        for itr in range(self.n_iter):
            for coef in range(self.n_coefs):
                schedule += [("sweep", itr, coef, step) for step in range(self.n_steps)]
                schedule.append(("check", itr, coef, None))
        return schedule

    def pre_func_signal(self):
        """Prepare the signal"""
        # initialize the mirror and coef lists, etc

        # Timing
        self.start_time = time.time()

        self.done_all = False
        self.done_itr = False

        self.n_iter = self.tw_settings["iterations"]
        self.n_steps = self.tw_settings["steps"]
        self.coef_amp = self.tw_settings["amplitude"]

        self.coef_sweep = np.linspace(
            -self.coef_amp, self.coef_amp, self.n_steps
        ).astype(np.float32)

        self.signal_id = 0
        self.schedule = self.build_schedule()
        self.total_frame_num = len(self.schedule)

        # Hand-shake with the data thread. The signal thread waits on these so
        # that every sweep frame is taken with all accepted corrections applied.
        #: threading.Condition: guards fits_done and decisions_done.
        self.progress = threading.Condition()
        #: int: sweeps fitted so far (set by the data thread).
        self.fits_done = 0
        #: int: check frames decided so far (set by the data thread).
        self.decisions_done = 0
        #: np.ndarray: accepted coefficients plus the last fitted update.
        self.candidate_coefs = None
        #: bool: the run was stopped by a timeout or repeated mirror failures.
        self.aborted = False
        self.mirror_failures = 0
        self.final_written = False

        if self.start_from == "flat":
            self.best_coefs = np.zeros(self.n_modes, dtype=np.float32)
        elif self.start_from == "current":
            curr_expt_coefs = list(
                self.model.configuration["experiment"]["MirrorParameters"][
                    "modes"
                ].values()
            )
            self.best_coefs = np.asarray(curr_expt_coefs, dtype=np.float32)

        # Per-mode sweep scale: 1.0 unless the mirror reports amplitude limits
        # (Phaseform DPP); then each sweep stays within that mode's range.
        self.sweep_scale = np.ones(self.n_coefs, dtype=np.float32)
        try:  # mirrors without amplitude limits (e.g. Imagine Optics) raise here
            pos_lim, neg_lim = self.mirror_controller.get_amplitude_limits()
        except (AttributeError, TypeError, ValueError):
            pos_lim = neg_lim = None
        if pos_lim is not None and self.coef_amp > 0:
            for j, c in enumerate(self.change_coef):
                room = SWEEP_LIMIT_MARGIN * min(
                    pos_lim[c] - self.best_coefs[c], self.best_coefs[c] - neg_lim[c]
                )
                self.sweep_scale[j] = np.clip(
                    room / self.coef_amp, SWEEP_SCALE_MIN, 1.0
                )
                if room < SWEEP_SCALE_MIN * self.coef_amp:
                    self.model.logger.warning(
                        f"TonyWilson: mode {self.mode_names[c]} has only "
                        f"{max(room, 0.0):.3f} of range left; its sweep will "
                        "saturate."
                    )
            self.model.logger.info(
                f"TonyWilson sweep scale per mode: {self.sweep_scale}"
            )

        #: np.ndarray: accepted coefficients (kept for compatibility).
        self.best_coefs_overall = deepcopy(self.best_coefs)
        self.best_metric = 0.0
        self.best_peaks = []

        self.model.logger.info(
            "TonyWilson: start; modes "
            f"{[self.mode_names[c] for c in self.change_coef]}, iterations "
            f"{self.n_iter}, steps {self.n_steps}, amplitude {self.coef_amp}, "
            f"fit {self.fit_func}, metric {self.metric}, from {self.start_from}, "
            f"{self.total_frame_num} frames, check tolerance {CHECK_TOLERANCE:.0%}"
        )

    def wait_for(self, counter, target):
        """Block the signal thread until the data thread has done `target` fits or
        decisions (`counter` is "fits_done" or "decisions_done").

        Returns
        -------
        bool
            False if acquisition stopped, or on timeout (the run is then
            aborted), True otherwise.
        """
        deadline = time.monotonic() + WAIT_TIMEOUT_S
        with self.progress:
            while getattr(self, counter) < target:
                if self.model.stop_acquisition or self.aborted:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.abort(
                        f"the data thread did not finish a {counter} within "
                        f"{WAIT_TIMEOUT_S:g} s"
                    )
                    return False
                self.progress.wait(min(remaining, 0.1))
        return True

    def abort(self, reason):
        """Stop the run and leave the mirror at the accepted coefficients."""
        if self.aborted:
            return
        self.aborted = True
        self.model.logger.error(f"TonyWilson: stopping the run: {reason}.")
        try:
            self.mirror_controller.display_modes(self.best_coefs)
            self.final_written = True
            self.model.logger.info(
                "TonyWilson: mirror set to the accepted coefficients (run aborted)."
            )
        except Exception as e:  # noqa: BLE001 - the run is ending anyway
            print(e)

    def in_func_signal(self):
        """Run the signal.

        Returns
        -------
        bool
            True if the signal is done, False otherwise
        """
        if self.aborted or self.signal_id >= self.total_frame_num:
            return True

        kind, itr, coef, step = self.schedule[self.signal_id]
        sweep_index = None if itr is None else itr * self.n_coefs + coef

        # A sweep starts only after the previous check was decided, and a check
        # needs its sweep's fit, so no frame is taken on a stale shape.
        if (
            kind == "sweep"
            and step == 0
            and not self.wait_for("decisions_done", sweep_index)
        ):
            return True
        if kind == "check" and not self.wait_for("fits_done", sweep_index + 1):
            return True

        out_str = "in_func_signal\n"
        out_str += f"\tSignal:\t{self.signal_id} ({kind})\n"
        out_str += f"\tStep:\t{step}\n"
        out_str += f"\tC_n:\t{coef}\n"
        out_str += f"\tItr:\t{itr}\n"

        if kind == "check":
            applied_coefs = self.candidate_coefs.copy()
        else:
            coef_arr = np.zeros(self.n_modes, dtype=np.float32)
            if kind == "sweep":
                c = self.change_coef[coef]
                coef_arr[c] = self.coef_sweep[step] * self.sweep_scale[coef]
            applied_coefs = coef_arr + self.best_coefs

        out_str += f"\tApply:\t[{' '.join([f'{c:.2f}' for c in applied_coefs])}]\n"

        # Update the mirror...
        self.mirror_controller.flat()
        try:
            self.mirror_controller.display_modes(applied_coefs)
        except Exception as e:
            print(e)
            self.count_mirror_failure()
            return self.aborted

        time.sleep(self.model.active_microscope.current_exposure_time / 1000)

        try:
            curr_mirror_coefs = self.mirror_controller.get_modal_coefs()[0]

            out_str += (
                f"\tCoefs:\t[{' '.join([f'{c:.2f}' for c in curr_mirror_coefs])}]\n"
            )
        except Exception as e:
            print(e)
            self.count_mirror_failure()
            return self.aborted

        self.model.logger.debug(
            f"*** TonyWilson > in_func_signal :: {kind} iter:{itr}\tcoef:{coef}"
            f"\tstep:{step}"
        )
        coef_str = " ".join([f"{c:.2f}" for c in applied_coefs])
        self.model.logger.debug(
            f"*** TonyWilson > in_func_signal :: display_modes:[{coef_str}]"
        )

        if (applied_coefs == curr_mirror_coefs).all() or (applied_coefs == 0).all():
            self.mirror_failures = 0
            self.signal_id += 1
            self.tw_frame_queue.put((self.model.frame_id, kind, itr, coef, step))
        else:
            out_str += "\tMirror update failed...\n"
            self.count_mirror_failure()

        if self.verbose:
            print(out_str)

        return self.signal_id >= self.total_frame_num or self.aborted

    def count_mirror_failure(self):
        """Abort after MAX_MIRROR_FAILURES failed mirror updates in a row (the
        same frame would otherwise be retried forever)."""
        self.mirror_failures += 1
        if self.mirror_failures >= MAX_MIRROR_FAILURES:
            self.abort(f"{self.mirror_failures} mirror updates in a row failed")

    def end_func_signal(self):
        """End the signal

        Returns
        -------
        bool
            True if the signal is done, False otherwise
        """
        if self.model.stop_acquisition:
            return True

        return self.signal_id >= self.total_frame_num or self.done_all or self.aborted

    def pre_func_data(self):
        """Prepare the data"""
        #: tuple: queued frame (frame_id, kind, itr, coef, step) whose image has
        #: not arrived yet; kept on self because it can arrive in a later call.
        self.pending = None
        self.get_frames_num = 0
        self.plot_data = []
        self.trace_list = {}
        self.total_frame_num = self.get_tw_frame_num()
        self.x = self.coef_sweep
        self.y = []
        self.x_fit = np.linspace(-self.coef_amp, self.coef_amp, 1024)
        self.y_fit = []
        self.mirror_img = None

        self.frames_done = 0

        #: float: metric of the frame taken with the accepted coefficients.
        self.reference_metric = None
        #: list: one entry per check frame (for the log and the summary).
        self.decisions = []

    def process_data(self, coef, mode="poly"):
        """Fit the sweep of armed mode `coef` (index into change_coef).

        Parameters
        ----------
        coef : int
            Armed-mode index of the sweep just completed
        mode : str, optional
            Fitting mode, by default "poly"

        Returns
        -------
        tuple
            (update, R^2): the R^2-weighted correction for this mode, and R^2.
        """
        self.y = self.plot_data

        try:
            r_2 = self._fit(mode)
        except (RuntimeError, ValueError) as e:  # curve_fit did not converge
            self.model.logger.warning(
                f"TonyWilson: fit failed ({e}); this mode is left unchanged."
            )
            self.y_fit = np.zeros_like(self.x_fit)
            r_2 = 0.0
        if not np.isfinite(r_2):
            # e.g. identical metrics at every step (0/0): no information, and a
            # NaN would make every later command fail the readback check.
            self.model.logger.warning(
                "TonyWilson: the metric did not change across the sweep; this "
                "mode is left unchanged."
            )
            r_2 = 0.0

        # weight by R^2 goodness of fit
        update = self.x_fit[self.y_fit.argmax()] * r_2 * self.sweep_scale[coef]
        self.mirror_img = self.mirror_controller.get_wavefront_pix()

        # Plot data, as before: the middle frame of each sweep, and the trace
        # (now stored under the name of the mode that was swept).
        self.best_peaks.append(self.plot_data[int(self.n_steps / 2)])
        self.trace_list[self.mode_names[self.change_coef[coef]]] = {
            "x": np.array(self.x, copy=True),
            "y": self.y,
            "x_fit": self.x_fit[::32],
            "y_fit": self.y_fit[::32],
        }

        self.plot_data = []
        return update, r_2

    def _fit(self, mode):
        """Fit the sweep with the chosen function; return R^2 (sets self.y_fit)."""
        if mode == "poly":
            c = np.min(self.y)  # offset guess
            b = (np.max(self.y) - c) / self.coef_amp  # slope guess
            a = -b / 2

            p, _ = curve_fit(
                poly2,
                self.x,
                self.y,
                p0=[a, b, c],
                bounds=([-np.inf, -np.inf, -np.inf], [0.0, np.inf, np.inf]),
            )
            self.y_fit = poly2(self.x_fit, p[0], p[1], p[2])
            r_2 = r_squared(self.y, poly2(self.x, p[0], p[1], p[2]))

        elif mode == "gauss":
            d = np.min(self.y)
            a = np.max(self.y) - d
            b = self.x[np.argmax(self.y)]
            c = (self.x[-1] - self.x[0]) / 2

            p, _ = curve_fit(
                gauss,
                self.x,
                self.y,
                p0=[a, b, c, d],
                bounds=([0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
            )
            self.y_fit = gauss(self.x_fit, p[0], p[1], p[2], p[3])
            r_2 = r_squared(self.y, gauss(self.x, p[0], p[1], p[2], p[3]))

        return r_2

    def measure(self, img):
        """Return the selected image metric."""
        if self.metric == "Pixel Max":
            return img.max()
        elif self.metric == "Pixel Average":
            return img.mean()
        elif self.metric == "DCT Shannon Entropy":
            return img_contrast.fast_normalized_dct_shannon_entropy(img, 3)[0]

    def fit_sweep(self, itr, coef):
        """Fit the completed sweep and propose accepted + update for checking."""
        c = self.change_coef[coef]
        metrics = list(self.plot_data)
        update, r_2 = self.process_data(coef, mode=self.fit_func)
        candidate = self.best_coefs.copy()
        candidate[c] += update
        self.candidate_coefs = candidate
        self.model.logger.info(
            f"TonyWilson: iteration {itr + 1}/{self.n_iter}, mode "
            f"{self.mode_names[c]}: sweep metrics {[float(m) for m in metrics]}, "
            f"fit update {float(update):+.4f} (R^2 {float(r_2):.3f}), candidate "
            f"{float(candidate[c]):+.4f}"
        )
        with self.progress:
            self.fits_done += 1
            self.progress.notify_all()

    def decide(self, metric, itr, coef):
        """Keep or undo the last update, judged by the check frame."""
        c = self.change_coef[coef]
        name = self.mode_names[c]
        reference = self.reference_metric
        threshold = reference - CHECK_TOLERANCE * abs(reference)
        accepted = bool(metric >= threshold)
        if accepted:
            self.best_coefs = self.candidate_coefs.copy()
            self.reference_metric = metric
            self.model.logger.info(
                f"TonyWilson: iteration {itr + 1}/{self.n_iter}, mode {name}: check "
                f"{float(metric):.6g} vs reference {float(reference):.6g} -> "
                f"accepted, coefficient now {float(self.best_coefs[c]):+.4f}"
            )
        else:
            self.model.logger.warning(
                f"TonyWilson: iteration {itr + 1}/{self.n_iter}, mode {name}: check "
                f"{float(metric):.6g} is more than {CHECK_TOLERANCE:.0%} below the "
                f"reference {float(reference):.6g} -> update "
                f"{float(self.candidate_coefs[c] - self.best_coefs[c]):+.4f} "
                f"rejected, coefficient stays {float(self.best_coefs[c]):+.4f}"
            )
        self.best_coefs_overall = deepcopy(self.best_coefs)
        self.decisions.append(
            {"iteration": itr, "mode": name, "accepted": accepted, "check": metric}
        )

        if coef == self.n_coefs - 1:
            # Shrink the sweep before the signal thread is released.
            self.coef_sweep *= 0.95
            self.done_itr = True
            print(f"Done iteration {itr + 1} / {self.n_iter} ...")
            if itr == self.n_iter - 1:
                self.done_all = True

        with self.progress:
            self.decisions_done += 1
            self.progress.notify_all()

    def in_func_data(self, frame_ids=[]):
        """Run the data

        Parameters
        ----------
        frame_ids : list, optional
            List of frame ids, by default []

        Returns
        -------
        list
            List of frame ids
        """
        self.get_frames_num += len(frame_ids)

        out_str = "in_func_data\n"
        out_str += f"\tFrames:\t{frame_ids}\n"
        out_str += f"\tGet Frames Num:\t{self.get_frames_num}\n"

        while True:
            if self.pending is None:
                try:
                    self.pending = self.tw_frame_queue.get_nowait()
                except Exception:
                    out_str += "\tStill waiting tw_frame_queue...\n"
                    break
            frame_id, kind, itr, coef, step = self.pending
            if frame_id not in frame_ids:
                out_str += f"\tFrame ID {frame_id} was not in the frame queue...\n"
                break
            self.pending = None

            new_data = self.measure(self.model.data_buffer[frame_id])
            self.frames_done += 1
            out_str += f"\t{kind}: {new_data}\tDone:\t{self.frames_done}\n"

            if kind == "reference":
                self.reference_metric = new_data
                self.model.logger.info(f"TonyWilson: reference metric {new_data:.6g}")
            elif kind == "sweep":
                self.plot_data.append(new_data)
                self.model.logger.debug(
                    f"*** TonyWilson > in_func_data :: plot_data: {self.plot_data}"
                )
                if step == self.n_steps - 1:
                    self.fit_sweep(itr, coef)
            else:
                self.decide(new_data, itr, coef)

        self.tw_data_queue.put((self.frames_done,))

        if self.verbose:
            print(out_str)

        if self.frames_done >= self.total_frame_num:
            return frame_ids

    def build_report(self):

        print("**** best coefs after all iterations:", self.best_coefs_overall)

        return deepcopy(
            {
                "mirror_update": {
                    "mirror_img": self.mirror_img,
                    "coefs": self.best_coefs,
                },
                "tonywilson": {
                    "peaks": self.best_peaks,
                    "trace": self.trace_list,
                    "done": self.done_all,
                    "metric": self.metric,
                    "iter": self.n_iter,
                    "steps": self.n_steps,
                    "amp": self.coef_amp,
                    "modes": self.change_coef,
                },
            }
        )

    def end_func_data(self):
        """End the data

        Returns
        -------
        bool
            True if the data is done, False otherwise
        """
        if self.done_all:
            try:
                stop_time = time.time()
                print(f"Total runtime:\t{(stop_time - self.start_time):.3f} sec")
                rejected = [
                    (d["iteration"] + 1, d["mode"])
                    for d in self.decisions
                    if not d["accepted"]
                ]
                self.model.logger.info(
                    f"TonyWilson: done in {stop_time - self.start_time:.1f} s, "
                    f"{self.frames_done} frames, "
                    f"{len(self.decisions) - len(rejected)}/{len(self.decisions)} "
                    f"updates accepted, rejected (iteration, mode): {rejected}; "
                    "final coefficients "
                    + str(
                        {
                            self.mode_names[c]: round(float(self.best_coefs[c]), 4)
                            for c in self.change_coef
                        }
                    )
                )
            except Exception as e:
                print(e)

        try:
            if self.done_itr:
                current_report = self.build_report()
                self.report.append(current_report)

                self.model.event_queue.put(
                    ("mirror_update", current_report["mirror_update"])
                )
                self.model.event_queue.put(("tonywilson", current_report["tonywilson"]))
        except Exception as e:
            print(e)

        self.done_itr = False

        # Apply the accepted shape once, when the run ends or is stopped. During
        # the run the signal thread owns the mirror: a write here could land in
        # the exposure of the next frame.
        finished = self.done_all or self.aborted or self.model.stop_acquisition
        if finished and not self.final_written:
            self.final_written = True
            self.mirror_controller.display_modes(self.best_coefs)
            self.model.logger.info(
                "TonyWilson: mirror set to the accepted coefficients ("
                + ("run complete" if self.done_all else "run stopped")
                + ")."
            )

        if self.done_all and self.save_report:
            self.model.event_queue.put(("ao_save_report", self.report))

        return self.frames_done >= self.total_frame_num or self.done_all or self.aborted
