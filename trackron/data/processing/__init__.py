from .base import  SiameseBaseProcessing, SiamProcessing
from .dimp_processing import DiMPProcessing, KLDiMPProcessing
from .seq_processing import SequenceProcessing
from .vos_processing import VOSProcessing
from .stark_processing import StarkProcessing
from .siamrpn_processing import SiamRPNProcessing
from .mot_processing import MOTProcessing
from .build import build_processing_class

__all__ = list(globals().keys())
