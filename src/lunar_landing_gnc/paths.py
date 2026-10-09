"""Where external data lives.

The LOLA DEM, the LROC NAC frames and the SPICE kernels are not in this
repository. They are read from a lunar-polar-traverse data directory named by
the environment variable LLG_E_DATA, which holds lola/, nac/ and kernels/.
Nothing here names an absolute path; with the variable unset, code that needs
the data raises and tests that need it skip.
"""
import os
from pathlib import Path

ENV_VAR = "LLG_E_DATA"

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "data" / "reference"
LROC_OBSERVATIONS = REFERENCE / "lroc_nac_observations.csv"

# The kernel set carried from lunar-polar-traverse. moon_080317.tf defines
# MOON_ME_DE421; moon_pa_de421_1900-2050.bpc orients it.
KERNELS = (
    "naif0012.tls",
    "pck00010.tpc",
    "moon_pa_de421_1900-2050.bpc",
    "moon_080317.tf",
    "de421.bsp",
)


def e_data():
    """The external data directory, or None when LLG_E_DATA is unset."""
    value = os.environ.get(ENV_VAR)
    return Path(value) if value else None


def require_e_data():
    """The external data directory; raises if LLG_E_DATA is unset or wrong."""
    root = e_data()
    if root is None:
        raise RuntimeError(f"{ENV_VAR} is not set; point it at lunar-polar-traverse/data")
    if not root.is_dir():
        raise RuntimeError(f"{ENV_VAR}={root} is not a directory")
    return root


def kernel_paths(root=None):
    """Full paths of the kernel set, in load order."""
    root = require_e_data() if root is None else Path(root)
    return [root / "kernels" / name for name in KERNELS]


def nac_path(product, kind, root=None):
    """A NAC GeoTIFF: kind is "LE", "RE" or "MODEL"."""
    root = require_e_data() if root is None else Path(root)
    suffix = "_20MPP_MODEL.TIF" if kind == "MODEL" else f"{kind}_20MPP.TIF"
    return root / "nac" / f"{product}{suffix}"
