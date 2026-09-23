# syntax=docker/dockerfile:1

# ---- build : installe les dependances gelees, jetable -----------------
FROM python:3.13-slim AS build
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# --no-editable : sans ca, le paquet installe pointerait vers ./src au lieu
# d'un wheel fige et autonome dans .venv - l'etape runtime ne recopie pas
# ./src, donc l'API ne demarrerait pas sans ce flag.
# --no-dev : le groupe dev (dvc, streamlit, locust, pytest, black, ruff) est
# installe par defaut par uv sync ; inutile a l'API, il gonflait l'image.
RUN uv sync --frozen --no-editable --no-dev

# ---- runtime : repart d'une image propre, non-root ---------------------
FROM python:3.13-slim AS runtime
WORKDIR /app

RUN groupadd --system appgroup \
    && useradd --system --uid 10001 --gid appgroup --no-create-home appuser

COPY --from=build /app/.venv /app/.venv
COPY artifacts/models/model.joblib /app/artifacts/models/model.joblib

ENV PATH="/app/.venv/bin:${PATH}" \
    MODEL_PATH="/app/artifacts/models/model.joblib" \
    PREFECT_HOME="/app/data/.prefect"

# /app est possede par root (COPY ci-dessus) : le flow (module 30) ecrit
# data/gold/gold_dataset.csv et son etat Prefect (PREFECT_HOME) en cours de
# route. Seuls ces sous-repertoires passent a appuser, pas /app entier -
# .venv et le modele restent en lecture seule pour l'utilisateur applicatif.
RUN mkdir -p /app/data /app/artifacts && chown -R appuser:appgroup /app/data /app/artifacts

USER appuser
EXPOSE 8000

# Pas de curl/wget dans python:3.13-slim (les ajouter grossirait l'image) :
# la sonde utilise le stdlib deja present dans .venv.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "indusense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
