import sys
from pathlib import Path

here = Path(__file__)
sys.path.append(str(here.parent.parent))

import frontend

frontend.main()