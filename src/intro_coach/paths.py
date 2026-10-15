import os
from pathlib import Path

HOME = Path(os.environ.get("INTRO_COACH_HOME", Path.home() / ".intro-coach"))
MODELS = HOME / "models"
TAKES = HOME / "takes"
