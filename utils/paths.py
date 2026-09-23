"""Resolution of external asset locations (Wan2.1 base models, CUT3R).

Both can be set per-run on the command line (``--wan_root`` / ``--cut3r_root``)
or through the environment (``MVF_WAN_ROOT`` / ``MVF_CUT3R_ROOT``). Defaults are
relative to the repository root, matching the layout in the README:

    wan_models/Wan2.1-T2V-1.3B/
    CUT3R/src/
"""

import os
import sys

WAN_ROOT_ENV = "MVF_WAN_ROOT"
CUT3R_ROOT_ENV = "MVF_CUT3R_ROOT"

DEFAULT_WAN_ROOT = "wan_models"
DEFAULT_CUT3R_ROOT = "CUT3R"
DEFAULT_WAN_MODEL = "Wan2.1-T2V-1.3B"


def set_wan_root(path):
    """Register the directory holding the Wan2.1 model folders."""
    if path:
        os.environ[WAN_ROOT_ENV] = path


def set_cut3r_root(path):
    """Register the CUT3R checkout directory."""
    if path:
        os.environ[CUT3R_ROOT_ENV] = path


def wan_root():
    return os.environ.get(WAN_ROOT_ENV, DEFAULT_WAN_ROOT)


def wan_model_dir(model_name=DEFAULT_WAN_MODEL):
    """Directory of a Wan2.1 release, e.g. ``wan_models/Wan2.1-T2V-1.3B``."""
    return os.path.join(wan_root(), model_name)


def wan_file(relative_path, model_name=DEFAULT_WAN_MODEL):
    """A file inside a Wan2.1 release, e.g. ``Wan2.1_VAE.pth``."""
    return os.path.join(wan_model_dir(model_name), relative_path)


def cut3r_root():
    return os.environ.get(CUT3R_ROOT_ENV, DEFAULT_CUT3R_ROOT)


def add_cut3r_to_path(root=None):
    """Put CUT3R's ``src`` directory on ``sys.path`` so ``dust3r`` is importable."""
    src = os.path.join(root or cut3r_root(), "src")
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"CUT3R source not found at '{src}'. Clone CUT3R and pass "
            f"--cut3r_root /path/to/CUT3R (or set {CUT3R_ROOT_ENV})."
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def load_cut3r(weights_path, device, root=None):
    """Import CUT3R lazily and load a frozen model in eval mode."""
    add_cut3r_to_path(root)
    from dust3r.model import ARCroco3DStereo  # noqa: E402  (needs sys.path above)

    model = ARCroco3DStereo.from_pretrained(weights_path).to(device).eval()
    model.requires_grad_(False)
    return model
