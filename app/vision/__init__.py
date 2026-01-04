# app/vision/__init__.py

# =========================
# v4: ROI extraction
# =========================
from .roi import RoiExtractor, RoiResult


# =========================
# v4: Boundary detection (current default)
# =========================
from .boundary_detection_v4 import (
    BoundaryDetectorV4,
    BoundaryV4Config,
    BoundaryV4Result,
)

# Public alias (hide version)
BoundaryDetector = BoundaryDetectorV4
BoundaryConfig = BoundaryV4Config
BoundaryResult = BoundaryV4Result


# =========================
# v5: Tick detection (current default)
# =========================
from .ticks import (
    TickDetector as _TickDetectorV5,
    TickDetectionConfig,
    TickCandidate,
    TickSet,
)

# Public alias (hide implementation detail)
TickDetector = _TickDetectorV5


# =========================
# v5: PPM mapping
# =========================
from .ppm import (
    compute_ppm_from_ticks,
    PpmResult,
    PpmMappingConfig,
)


__all__ = [
    # ROI
    "RoiExtractor",
    "RoiResult",

    # Boundary (version-hidden)
    "BoundaryDetector",
    "BoundaryConfig",
    "BoundaryResult",

    # Tick detection (version-hidden)
    "TickDetector",
    "TickDetectionConfig",
    "TickCandidate",
    "TickSet",

    # PPM mapping
    "compute_ppm_from_ticks",
    "PpmResult",
    "PpmMappingConfig",
]
