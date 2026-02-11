import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generic.experiments.grid_search.primal_dual_grid_search import main

if __name__ == "__main__":
    main()
