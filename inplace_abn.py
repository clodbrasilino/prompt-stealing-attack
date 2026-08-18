# Backward-compatibility shim: re-export from src/ for scripts that run
# from the repo root (where src/ is not explicitly on sys.path).
from src.inplace_abn import InPlaceABN, ABN  # noqa: F401
