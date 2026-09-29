"""Test configuration.

PyTorch and XGBoost each ship an OpenMP runtime. On macOS, loading both in one process with several threads can
crash or hang, so the tests run single-threaded. This must be set before either library is imported.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
