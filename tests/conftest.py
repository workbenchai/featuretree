"""Put the featuretree package root on sys.path so `import ir` / `import step_recognize` work when
pytest is run from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def has_freecad():
    """Whether runner can find a freecadcmd here (AppImage, cache, or FREECAD_CMD — e.g. a macOS
    FreeCAD.app). The FreeCAD parity tests skip without one."""
    try:
        import runner
        runner.freecadcmd_path()
        return True
    except Exception:
        return False
