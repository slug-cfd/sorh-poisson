"""Definitions and coefficient models for nonlinear problems."""

from .coefficients import build_coefficient, ellipticity_ratio
from .registry import get_nonlinear_case, list_nonlinear_cases

__all__ = [
    "build_coefficient",
    "ellipticity_ratio",
    "get_nonlinear_case",
    "list_nonlinear_cases",
]
