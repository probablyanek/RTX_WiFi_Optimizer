from __future__ import annotations

import math

SPEED_OF_LIGHT_M_S = 299_792_458.0


def fspl_db(distance_m: float, frequency_hz: float) -> float:
    """Free-space path loss in dB for a distance and carrier frequency."""
    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz
    distance_m = max(float(distance_m), 1e-9)
    return 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength_m)


def db10(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1e-300))
