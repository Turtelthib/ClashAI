# clashai/adb/exceptions.py
# Exception hierarchy for the ADB layer.
#
# Catchers should prefer the most specific subclass. Code that wants to
# treat any ADB failure as recoverable catches the base ADBError.

# Inherit from the project-wide ClashAIError so a top-level handler can
# catch every ClashAI failure with one except. ClashAIError lives in the
# leaf module clashai/_core_exceptions.py to avoid a circular import with
# clashai/exceptions.py (which re-exports our classes).
from clashai._core_exceptions import ClashAIError


class ADBError(ClashAIError):
    """Base for any ADB failure. Catch this to recover from arbitrary
    ADB I/O problems."""


class ADBNotFoundError(ADBError):
    """The `adb` executable is not on PATH. Usually means Android Platform
    Tools aren't installed or PATH isn't set."""


class ADBTimeoutError(ADBError):
    """An adb call exceeded its timeout. The emulator might be frozen, or
    the host overloaded."""


class ADBNotConnectedError(ADBError):
    """`adb devices` doesn't list the configured device. Either the
    emulator isn't running or its serial doesn't match ADB_DEVICE
    (`localhost:6520` by default)."""
