import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bgap.experiments.grid_search.util_pricing_grid_search import main

if __name__ == "__main__":
    main()
