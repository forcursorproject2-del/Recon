import sys
import os
from pathlib import Path

# Add project root directory to sys.path for imports to work correctly in tests
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
