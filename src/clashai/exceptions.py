# clashai/exceptions.py
# Project-wide exception hierarchy.
#
# Catch the most specific subclass that makes sense. Generic recovery
# (e.g. "anything from the ADB layer is recoverable, retry") catches the
# corresponding domain root (ADBError). Catching ClashAIError catches
# every project-defined exception — useful for the brain's top-level
# guard that prevents a single agent crash from killing the orchestrator.
#
# Sub-domains can have their own module (clashai/adb/exceptions.py does)
# and re-parent here. The cross-import order is:
#   1. ClashAIError defined below
#   2. clashai.adb.exceptions then imports ClashAIError to parent its tree
#   3. We re-export the ADB subclasses for convenience


# Root --------------------------------------------------------------------
# Defined in clashai/_core_exceptions.py so the leaf sub-packages
# (e.g. clashai.adb.exceptions) can import it without a circular
# dependency on this re-export module.

from clashai._core_exceptions import ClashAIError


# Perception --------------------------------------------------------------

class PerceptionError(ClashAIError):
    """Vision / model inference failures."""


class ScreenCaptureError(PerceptionError):
    """Couldn't grab a frame (WGC dead, ADB screencap failed, window not
    found, …)."""


class ModelInferenceError(PerceptionError):
    """A YOLO / CNN forward pass raised, the model wasn't loaded, or the
    output couldn't be parsed."""


# Navigation --------------------------------------------------------------

class NavigationError(ClashAIError):
    """Stuck while trying to reach a target screen state."""


class ScreenStateError(NavigationError):
    """The screen-state CNN's prediction doesn't match what the agent
    expected, with low confidence to back it up."""


# Combat ------------------------------------------------------------------

class CombatError(ClashAIError):
    """Failures during the battle phase."""


class TroopDeployError(CombatError):
    """A deploy action didn't go through (slot empty, troop not in bar,
    tap missed the deploy zone, …)."""


# Agent / orchestrator ---------------------------------------------------

class AgentError(ClashAIError):
    """Errors raised by sub-agents (V5.1+)."""


class AgentDisabledError(AgentError):
    """An agent refuses to run (can_run() returned False, or it's
    currently on cooldown). Used as a signal, not an error per se."""


# ADB --------------------------------------------------------------------
# Re-exported from clashai/adb/exceptions.py. That module imports
# ClashAIError (already defined above) and parents its own tree under it.

from clashai.adb.exceptions import (  # noqa: E402
    ADBError,
    ADBNotFoundError,
    ADBTimeoutError,
    ADBNotConnectedError,
)
