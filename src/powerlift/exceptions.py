"""Exception hierarchy for powerlift.

Every error raised on purpose by the library derives from
:class:`PowerliftError`, so callers can catch library errors with a single
``except`` clause. The concrete classes also inherit from the closest
built-in exception (``ValueError`` / ``RuntimeError``) so that code written
against the standard conventions keeps working.
"""

from __future__ import annotations


class PowerliftError(Exception):
    """Base class for all errors raised deliberately by powerlift."""


class InvalidInputError(PowerliftError, ValueError):
    """An argument is out of range, has the wrong type or shape, or is inconsistent."""


class InsufficientDataError(PowerliftError, ValueError):
    """The data cannot support the requested computation (for example zero variance)."""


class ConvergenceError(PowerliftError, RuntimeError):
    """A numerical solver could not find a solution."""
