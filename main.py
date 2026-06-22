from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "project" / "src" / "main.py"), run_name="__main__")
