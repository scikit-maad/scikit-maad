from pathlib import Path

_ROOT_PATH: Path = Path(__file__).parent.parent

DATA_PATH: Path = _ROOT_PATH / 'data'
TESTS_DATA_PATH: Path = _ROOT_PATH / 'tests' / 'data'
