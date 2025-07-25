"""
Storage package for LateBench result management.
"""

from .critic_store import CriticResultStore, StoredResult, BatchResultSummary, get_default_store

__all__ = ['CriticResultStore', 'StoredResult', 'BatchResultSummary', 'get_default_store']