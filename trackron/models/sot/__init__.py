from .build import SOT_HEAD_REGISTRY, build_sot_head  # isort:skip

# import all the meta_arch, so they will be registered
from .s3t import S3THead, SiameseS3THead, SiameseS3TMultiQuery
from .df_detr import DeformableTransformerHead
from .token import TokenHead
from .decode import DecodeHead, SiameseDecodeHead, DecodeAllHead
from .stark import STARK
from .siamrpn import SiamRPN


__all__ = list(globals().keys())

