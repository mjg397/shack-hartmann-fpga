import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs
print("slopes_ref max:", shwfs.slopes_ref.max())
print("slopes_ref min:", shwfs.slopes_ref.min())
