"""Assemble les scripts 01 à 07 en un seul notebook Jupyter exécutable de bout en bout.
Usage : python python/build_notebook.py   ->   notebook/projet_KO_PEP.ipynb
Les scripts utilisent le format '# %%' : chaque '# %%' devient une cellule, '# %% [markdown]' une cellule texte."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["01_donnees_eda", "02_arima_sarima", "03_var_cointegration", "04_garch", "05_deep_learning", "06_evaluation", "07_llm_prompts"]


def cell(kind, text):
    src = text.strip("\n").splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": src}
    return {**base, "outputs": [], "execution_count": None} if kind == "code" else base


cells = [
    cell("markdown", "# Mini-projet Séries Temporelles — KO & PEP (2019-2024)\nExécuter toutes les cellules dans l'ordre. "
                     "Prérequis : `pip install -r requirements.txt` (Python 3.10-3.12 conseillé pour TensorFlow). Graine fixée : 42."),
    cell("code", "import sys\nfrom pathlib import Path\n\n"
                 "for project_dir in (Path.cwd(), *Path.cwd().parents):\n"
                 "    python_dir = project_dir / 'python'\n"
                 "    if (python_dir / 'common.py').is_file():\n"
                 "        sys.path.insert(0, str(python_dir))\n"
                 "        break\n"
                 "    if (project_dir / 'common.py').is_file():\n"
                 "        sys.path.insert(0, str(project_dir))\n"
                 "        break\n"
                 "else:\n"
                 "    raise FileNotFoundError('Impossible de trouver python/common.py depuis le dossier courant.')\n"
                 "from common import print_versions\nprint_versions()   # versions des packages documentées"),
]
for name in SCRIPTS:
    txt = (ROOT / "python" / f"{name}.py").read_text(encoding="utf-8")
    for block in re.split(r"(?m)^# %%", txt):
        if not block.strip():
            continue
        if block.startswith(" [markdown]"):
            md = "\n".join(re.sub(r"^# ?", "", l) for l in block.split("\n")[1:])
            cells.append(cell("markdown", md))
        else:
            cells.append(cell("code", block))
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
out = ROOT / "notebook" / "projet_KO_PEP.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(cells)} cellules écrites dans {out}")
