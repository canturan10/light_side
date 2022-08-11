import time
import os

from light_side.version import __version__ as pkg_version

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
__version__ = pkg_version

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__description__",
    "__homepage__",
    "__license__",
    "__license_url__",
    "__long_description__",
    "__pkg_name__",
    "__version__",
]
