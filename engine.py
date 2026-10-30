"""Public compatibility facade for the modular SmartWealth engine."""
from __future__ import annotations
import validation as _validation
import projection as _projection
import allocation as _allocation
import explain as _explain
import planner as _planner

def _reexport(module):
    for name in module.__all__:
        globals()[name] = getattr(module, name)

for _module in (_validation, _projection, _allocation, _explain, _planner):
    _reexport(_module)

load_data = _validation.load_data

def run_all(data_dir=_validation.DATA_DIR):
    return _planner.run_all(data_dir, loader=load_data)

def export_outputs(data_dir=_validation.DATA_DIR, output_dir=_validation.OUTPUT_DIR):
    return _planner.export_outputs(data_dir, output_dir, loader=load_data)

__all__ = [name for module in (_validation, _projection, _allocation, _explain, _planner) for name in module.__all__] + ["run_all", "export_outputs"]

