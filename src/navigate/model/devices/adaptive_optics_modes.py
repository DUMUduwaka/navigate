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

"""
Naviagte consider every wavefront corrector device (the Imagine Optics
deformable mirror and the Phaseform Deformable Phase plate) as a "mirror".

The two manufacturers use different Zernike mode ordering conventions,
    Imagine Optics: Wyant
    Phaseform: OSA/ANSI
"""


from typing import Optional

#: int: Number of Zernike modes tracked for AO correction.
DEFAULT_N_MODES = 32


#: List of the 32 Imagine Optics mirror Zernike mode names - Wyant ordering.
MIRROR_MODE_NAMES = [
    "Vert. Tilt",
    "Horz. Tilt",
    "Defocus",
    "Vert. Asm.",
    "Oblq. Asm.",
    "Vert. Coma",
    "Horz. Coma",
    "3rd Spherical",
    "Vert. Tre.",
    "Horz. Tre.",
    "Vert. 5th Asm.",
    "Oblq. 5th Asm.",
    "Vert. 5th Coma",
    "Horz. 5th Coma",
    "5th Spherical",
    "Vert. Tetra.",
    "Oblq. Tetra.",
    "Vert. 7th Tre.",
    "Horz. 7th Tre.",
    "Vert. 7th Asm.",
    "Oblq. 7th Asm.",
    "Vert. 7th Coma",
    "Horz. 7th Coma",
    "7th Spherical",
    "Vert. Penta.",
    "Horz. Penta.",
    "Vert. 9th Tetra.",
    "Oblq. 9th Tetra.",
    "Vert. 9th Tre.",
    "Horz. 9th Tre.",
    "Vert. 9th Asm.",
    "Oblq. 9th Asm.",
]


# List of the 32 Phaseform phase plate Zernike modes name - OSA ordering
PHASEFORM_MODE_NAMES = [
    "Vert. tilt",  # (n=1, m=-1)
    "Hor. tilt",  # (n=1, m=+1)
    "Obliq. astigm.",  # (n=2, m=-2)
    "Defocus",  # (n=2, m=0)
    "Vert. astigm.",  # (n=2, m=+2)
    "Vert. 3foil",  # (n=3, m=-3)
    "Vert. coma",  # (n=3, m=-1)
    "Hor. coma",  # (n=3, m=+1)
    "Obliq. 3foil",  # (n=3, m=+3)
    "Obliq. 4foil",  # (n=4, m=-4)
    "Obliq. 2d ast.",  # (n=4, m=-2)
    "Spherical",  # (n=4, m=0)
    "Vert. 2d ast.",  # (n=4, m=+2)
    "Vert. 4foil",  # (n=4, m=+4)
    "Vert. 5foil",  # (n=5, m=-5)
    "Vert. 2d 3foil",  # (n=5, m=-3)
    "Vert. 2d coma",  # (n=5, m=-1)
    "Hor. 2d coma",  # (n=5, m=+1)
    "Obliq. 2d 3foil",  # (n=5, m=+3)
    "Obliq. 5foil",  # (n=5, m=+5)
    "Obliq. 6foil",  # (n=6, m=-6)
    "Obliq.2d 4foil",  # (n=6, m=-4)
    "Obliq. 3d ast.",  # (n=6, m=-2)
    "2d spherical",  # (n=6, m=0)
    "Vert. 3d ast.",  # (n=6, m=+2)
    "Vert. 2d 4foil",  # (n=6, m=+4)
    "Vert. 6foil",  # (n=6, m=+6)
    "Vert. 7foil",  # (n=7, m=-7)
    "Vert. 2d 5foil",  # (n=7, m=-5)
    "Vert. 3d 3foil",  # (n=7, m=-3)
    "Vert. 3d coma",  # (n=7, m=-1)
    "Hor. 3d coma",  # (n=7, m=+1)
]

assert len(MIRROR_MODE_NAMES) == DEFAULT_N_MODES
assert len(PHASEFORM_MODE_NAMES) == DEFAULT_N_MODES
assert len(set(PHASEFORM_MODE_NAMES)) == DEFAULT_N_MODES

_MODE_NAMES_BY_MANUFACTURER = {
    "imop": MIRROR_MODE_NAMES,
    "dpp": PHASEFORM_MODE_NAMES,
}


def get_mirror_manufacturer(configuration: dict, microscope_name: str) -> Optional[str]:
    """Return the manufacturer key of a microscope's mirror hardware.

    Parameters
    ----------
    configuration : dict
        Global configuration (or any dict/DictProxy shaped like it).
    microscope_name : str
        Name of the microscope to check.

    Returns
    -------
    str or None
        ``"imop"`` or ``"dpp"`` (the module filename under
        ``devices/mirror/``, e.g. matching ``mirror/imop.py``/
        ``mirror/dpp.py``), or ``None`` if the microscope has no mirror
        configured or its hardware type can't be resolved.
    """
    try:
        hardware_type = configuration["configuration"]["microscopes"][microscope_name][
            "mirror"
        ]["hardware"]["type"]
    except (KeyError, TypeError):
        return None

    from navigate.config.device_schema import canonical_device_type

    canonical = canonical_device_type("mirror", hardware_type)
    if canonical is None or "." not in canonical:
        return None
    return canonical.split(".", 1)[0]


def get_active_mirror_manufacturer(configuration: dict) -> Optional[str]:
    """Return the mirror manufacturer key of the currently active microscope.

    Parameters
    ----------
    configuration : dict
        Global configuration (or any dict/DictProxy shaped like it).

    Returns
    -------
    str or None
        ``"imop"`` or ``"dpp"``, or ``None`` if it can't be resolved.
    """
    try:
        microscope_name = configuration["experiment"]["MicroscopeState"][
            "microscope_name"
        ]
    except (KeyError, TypeError):
        return None
    return get_mirror_manufacturer(configuration, microscope_name)


def get_mode_names(manufacturer: Optional[str]) -> list:
    """Return the 32 device-native Zernike mode names for a manufacturer.

    Parameters
    ----------
    manufacturer : str or None
        ``"imop"``, ``"dpp"``, or ``None``.

    Returns
    -------
    list
        A fresh copy of the 32-name list for that manufacturer, or an empty
        list if ``manufacturer`` is ``None``/unrecognized.
    """
    names = _MODE_NAMES_BY_MANUFACTURER.get(manufacturer)
    return list(names) if names else []
