"""
LateBench Dashboard
Web interface for analyzing mathematical reasoning errors
"""

from .app import app
from .utils import DashboardData

__version__ = "1.0.0"
__all__ = ["app", "DashboardData"]