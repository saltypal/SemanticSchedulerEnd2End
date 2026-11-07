"""Compatibility command; reusable implementation lives in Archived.owners_before_simplification_20261027.SemParam.training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Archived.owners_before_simplification_20261027.SemParam.training import main, train
if __name__ == '__main__':
    main()
