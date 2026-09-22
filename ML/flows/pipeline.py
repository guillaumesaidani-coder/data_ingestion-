"""Façade — la logique vit dans indusense.flows.predict_flow (module 30) :
le Dockerfile ne copie que src/ dans l'image, ce fichier-ci n'existe pas
dans le conteneur. Zéro logique dupliquée entre les deux points d'entrée.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.flows.predict_flow import indusense_pipeline, main  # noqa: F401

if __name__ == "__main__":
    main()
