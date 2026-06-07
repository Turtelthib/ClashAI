# clashai/adb/__init__.py
# ADB I/O layer for ClashAI.
#
# Single point of contact between Python code and the `adb` binary. All
# tap / swipe / key / screenshot operations route through ADBClient.
# Direct `subprocess.run(["adb", ...])` calls scattered across the codebase
# are being migrated to use this layer (Phase C.1+).

from clashai.adb.client import ADBClient, get_client
from clashai.adb.exceptions import (
    ADBError,
    ADBNotFoundError,
    ADBTimeoutError,
    ADBNotConnectedError,
)

__all__ = [
    'ADBClient', 'get_client',
    'ADBError', 'ADBNotFoundError',
    'ADBTimeoutError', 'ADBNotConnectedError',
]
