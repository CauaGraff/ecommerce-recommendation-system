import sys
from pathlib import Path

# Força o diretório raiz a entrar no sys.path do Python
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))