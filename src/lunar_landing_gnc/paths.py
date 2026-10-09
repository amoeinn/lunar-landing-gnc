"""Where external data lives.

The LOLA DEMs, the LROC NAC frames and the SPICE kernels are not in this
repository. They are read from a data directory named by the environment
variable LLG_DATA, holding lola/, nac/ and kernels/, which
examples/fetch_data.sh fills. Nothing here names an absolute path; with the
variable unset, code that needs the data raises and tests that need it skip.

LLG_DATA is read only. Everything this project computes goes to work/ in the
repository, which is not tracked.
"""
import os
from pathlib import Path

ENV_VAR = "LLG_DATA"

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "data" / "reference"
LROC_OBSERVATIONS = REFERENCE / "lroc_nac_observations.csv"
WORK = REPO_ROOT / "work"

# The kernel set carried from lunar-polar-traverse. moon_080317.tf defines
# MOON_ME_DE421; moon_pa_de421_1900-2050.bpc orients it.
KERNELS = (
    "naif0012.tls",
    "pck00010.tpc",
    "moon_pa_de421_1900-2050.bpc",
    "moon_080317.tf",
    "de421.bsp",
)


def data_dir():
    """The external data directory, or None when LLG_DATA is unset."""
    value = os.environ.get(ENV_VAR)
    return Path(value) if value else None


def require_data():
    """The external data directory; raises if LLG_DATA is unset or wrong."""
    root = data_dir()
    if root is None:
        raise RuntimeError(f"{ENV_VAR} is not set; see examples/fetch_data.sh")
    if not root.is_dir():
        raise RuntimeError(f"{ENV_VAR}={root} is not a directory")
    return root


def kernel_paths(root=None):
    """Full paths of the kernel set, in load order."""
    root = require_data() if root is None else Path(root)
    return [root / "kernels" / name for name in KERNELS]


def nac_path(product, kind, root=None):
    """A NAC GeoTIFF: kind is "LE", "RE" or "MODEL"."""
    root = require_data() if root is None else Path(root)
    suffix = "_20MPP_MODEL.TIF" if kind == "MODEL" else f"{kind}_20MPP.TIF"
    return root / "nac" / f"{product}{suffix}"
