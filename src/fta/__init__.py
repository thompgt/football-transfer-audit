"""Reusable pieces of the football transfer fairness audit.

The audit used to live entirely inside notebooks and two loose scripts at the
repository root, which made it impossible to test any of the joins or metrics.
Everything that carries an analytical claim now lives here and is exercised by
``tests/``.
"""

from fta import config  # noqa: F401

__all__ = ["config"]
