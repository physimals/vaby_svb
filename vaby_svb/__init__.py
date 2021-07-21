"""
Implementation of Stochastic Variational Bayesian inference for fitting timeseries data
"""
try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .svb import Svb
from .main import run

__all__ = [
   "__version__",
    "__timestamp__",
    "Svb",
    "run",
]
