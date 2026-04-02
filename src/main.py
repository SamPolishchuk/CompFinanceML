from pathlib import Path
import sys
base_dir  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

model_dir = base_dir / 'models'