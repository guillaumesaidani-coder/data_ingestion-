"""Configuration centralisée du pipeline vision (MVTec AD — bottle).

Aucun chemin ne doit être codé en dur ailleurs : notebooks et scripts
importent ce module plutôt que de répéter des `Path("...")` locaux.
"""

from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent
BOTTLE_ROOT     = ROOT_DIR / "bottle"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
FIGURES_DIR     = ROOT_DIR / "figures"
EMISSIONS_DIR   = ROOT_DIR / "emissions"
REPORTS_DIR     = ROOT_DIR / "reports"
NORM_STATS_PATH = ROOT_DIR / "norm_stats.json"

RANDOM_SEED      = 42
IMG_SIZE         = (256, 256)
VAL_RATIO        = 0.2
COUNTRY_ISO_CODE = "FRA"  # mix électrique forcé (salle/CI, indépendant de la géoloc IP)

DEFAULT_FILTERS    = (32, 64, 128, 64)  # ratio de compression ≈ 12 (retenu en TP3/TP4)
DEFAULT_ALPHA_LOSS = 0.8                # mse_ssim_loss : 0.8 x MSE + 0.2 x (1-SSIM)

for _d in (CHECKPOINTS_DIR, FIGURES_DIR, EMISSIONS_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)
