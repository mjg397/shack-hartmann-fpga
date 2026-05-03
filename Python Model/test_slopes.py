import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs
print("slopes_ref shape:", shwfs.slopes_ref.shape)
print("first few x refs:", shwfs.slopes_ref[0, :5])
