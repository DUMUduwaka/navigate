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


"""The Adaptive Optics menu opens one AO window, not a new one per click.

No Tk window and no controller are created: the popup and its controller are
replaced by mocks, so these tests do not touch the user's navigate config."""

import tkinter as tk
import types
from unittest import mock

import pytest

from navigate.controller.sub_controllers import menus


@pytest.fixture
def factories(monkeypatch):
    popup_cls = mock.MagicMock(name="AdaptiveOpticsPopup")
    controller_cls = mock.MagicMock(name="AdaptiveOpticsPopupController")
    monkeypatch.setattr(menus, "AdaptiveOpticsPopup", popup_cls)
    monkeypatch.setattr(menus, "AdaptiveOpticsPopupController", controller_cls)
    return popup_cls, controller_cls


def open_ao_menu(parent_controller):
    menu = types.SimpleNamespace(
        view=mock.sentinel.view, parent_controller=parent_controller
    )
    menus.MenuController.popup_adaptiveoptics(menu)


def existing_controller(window_exists):
    controller = mock.MagicMock(name="existing AO controller")
    if isinstance(window_exists, Exception):
        controller.view.popup.winfo_exists.side_effect = window_exists
    else:
        controller.view.popup.winfo_exists.return_value = window_exists
    return controller


def test_first_click_creates_the_window(factories):
    popup_cls, controller_cls = factories
    parent = types.SimpleNamespace(configuration={})

    open_ao_menu(parent)

    popup_cls.assert_called_once_with(mock.sentinel.view, configuration={})
    controller_cls.assert_called_once_with(popup_cls.return_value, parent)
    assert parent.ao_popup_controller is controller_cls.return_value


def test_second_click_brings_the_open_window_to_the_front(factories):
    popup_cls, controller_cls = factories
    existing = existing_controller(window_exists=1)
    parent = types.SimpleNamespace(configuration={}, ao_popup_controller=existing)

    open_ao_menu(parent)

    existing.showup.assert_called_once_with()
    popup_cls.assert_not_called()
    controller_cls.assert_not_called()
    assert parent.ao_popup_controller is existing


@pytest.mark.parametrize(
    "window_exists", [0, tk.TclError("bad window path name")], ids=["closed", "gone"]
)
def test_a_closed_window_is_created_again(factories, window_exists):
    popup_cls, controller_cls = factories
    existing = existing_controller(window_exists)
    parent = types.SimpleNamespace(configuration={}, ao_popup_controller=existing)

    open_ao_menu(parent)

    existing.showup.assert_not_called()
    controller_cls.assert_called_once_with(popup_cls.return_value, parent)
    assert parent.ao_popup_controller is controller_cls.return_value
