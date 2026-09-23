#!/usr/bin/env python
"""Mesure la taille réelle d'une image Docker construite et la compare à un
seuil (MAX_MB). Ne jamais se fier à la colonne "Size" de Docker Desktop :
avec le backend containerd, elle peut compter à tort un layer intermédiaire
de l'étape build comme s'il faisait partie de l'image livrée.
`docker inspect --format '{{.Size}}'` (confirmé par `docker save`) donne la
taille réelle de l'image exportée — c'est elle qu'on compare ici.
"""

import argparse
import subprocess
import sys

# Mesuré réellement (voir image_proof.md) avant de fixer ce seuil — jamais
# l'inverse : 690,9 Mo (docker inspect + docker save, confirmés identiques).
# Notre projet embarque tout uv.lock (jupyter/matplotlib/mlflow/optuna/...
# utiles aux notebooks et à l'entraînement, pas seulement à l'API) car ils
# sont dans les dépendances de base, pas dans un groupe séparé — les isoler
# casserait des notebooks qui comptent sur leur présence par défaut, hors
# scope de ce correctif (voir image_proof.md).
MAX_MB = 750


def get_image_size_bytes(image: str) -> int:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Size}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="indusense-api:m27")
    parser.add_argument("--max-mb", type=float, default=MAX_MB)
    args = parser.parse_args()

    size_bytes = get_image_size_bytes(args.image)
    size_mb = size_bytes / (1024 * 1024)
    print(f"{args.image} : {size_mb:.1f} Mo (seuil {args.max_mb:.0f} Mo)")

    if size_mb > args.max_mb:
        print("ECHEC : image au-dessus du seuil", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
