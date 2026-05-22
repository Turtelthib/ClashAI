# clashai/_core_exceptions.py
# Leaf module containing only the root exception class.
#
# Why a separate file: `clashai/exceptions.py` re-exports domain trees
# defined in their own packages (e.g. `clashai/adb/exceptions.py`). Those
# sub-trees inherit from ClashAIError. If ClashAIError lived in
# `clashai/exceptions.py`, the sub-tree would import it from there,
# creating a circular dependency. Hosting it in this no-deps leaf module
# breaks the cycle.

class ClashAIError(Exception):
    """Base for every project-defined exception."""
