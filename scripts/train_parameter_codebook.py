"""Compatibility command; reusable implementation lives in SemParam.training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from SemParam.training import main, train
if __name__ == '__main__':
    main()
