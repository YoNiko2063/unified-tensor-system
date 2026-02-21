"""
Root conftest.py — ensures the project root is on sys.path so that
top-level packages (tensor/, optimization/, etc.) are importable
without per-test sys.path manipulation.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
