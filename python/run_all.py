"""Exécute les scripts Python et construit le notebook et la présentation.
Le notebook généré doit ensuite être exécuté séparément pour valider ses cellules."""
import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for name in ["01_donnees_eda", "02_arima_sarima", "03_var_cointegration", "04_garch", "05_deep_learning", "06_evaluation",
             "07_llm_prompts", "build_notebook", "build_presentation"]:
    print(f"\n===================== {name} =====================")
    if subprocess.run([sys.executable, str(HERE / f"{name}.py")], cwd=HERE.parent).returncode != 0:
        sys.exit(f"Échec dans {name}.py : corriger puis relancer.")
print("\nTerminé. Étapes restantes : exécuter notebook/projet_KO_PEP.ipynb, "
      "Rscript R/classique.R, puis contrôler app/streamlit_app.py.")
