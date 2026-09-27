"""ctypes binding to NTIA's Irregular Terrain Model (Longley-Rice), point-to-point mode.

Built from https://github.com/NTIA/itm (C++; see scripts/build_itm.sh). pfl = [n_intervals, step_m, z0, z1, ...].
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

_LIB = None
ROOT = Path(__file__).resolve().parent.parent


def lib():
    global _LIB
    if _LIB is None:
        default = ROOT / "itm-src" / "build" / ("libitm.dylib" if sys.platform == "darwin" else "libitm.so")
        path = os.environ.get("ITM_LIB", str(default))
        _LIB = ctypes.CDLL(path)
        f = _LIB.ITM_P2P_TLS
        f.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                      ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                      ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_long)]
        f.restype = ctypes.c_int
    return _LIB


def p2p_loss(h_tx, h_rx, pfl, f_mhz, climate=5, n0=301.0, pol=1, eps=15.0, sigma=0.005, mdvar=12,
             time=50.0, location=50.0, situation=50.0):
    """Basic transmission loss (dB). climate 5 = continental temperate; pol 1 = vertical;
    mdvar 12 = single-message mode with direct situation variability (median values at 50/50/50)."""
    arr = np.ascontiguousarray(pfl, dtype=np.float64)
    a = ctypes.c_double()
    w = ctypes.c_long()
    rc = lib().ITM_P2P_TLS(h_tx, h_rx, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), climate, n0, f_mhz,
                           pol, eps, sigma, mdvar, time, location, situation, ctypes.byref(a), ctypes.byref(w))
    if rc not in (0, 1):   # 0 success, 1 success with warnings
        raise ValueError(f"ITM error code {rc}")
    return a.value, int(w.value)
