import time
import os

_PATH_ROOT = os.path.dirname(__file__)

_this_year = time.strftime("%Y")
__author__ = "Oguzcan Turan"
__author_email__ = "can.turan.10@gmail.com"
__copyright__ = f"Copyright (c) 2022-{_this_year}, {__author__}."
__description__ = (
    "PyTorch Lightning Implementations of Recent Low-Light Image Enhancement !"
)
__homepage__ = "https://github.com/canturan10/light_side"
__license__ = "MIT License"
__license_url__ = __homepage__ + "/blob/master/LICENSE"
__pkg_name__ = "light_side"


def _load_version(file_name: str = "version.py") -> str:
    """
    Load version from a py file.

    Args:
        file_name (str, optional): File name. Defaults to "version.py".

    Returns:
        str: Version.
    """
    # Open the file
    with open(os.path.join(_PATH_ROOT, file_name), "r", encoding="utf-8") as file:
        version = file.read().split("=")[-1].replace("'", "").replace('"', "").strip()
    return version


__version__ = _load_version()

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__description__",
    "__homepage__",
    "__license__",
    "__license_url__",
    "__pkg_name__",
    "__version__",
]
