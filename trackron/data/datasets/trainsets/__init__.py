from pathlib import Path
from trackron.utils.misc import jpeg4py_loader, opencv_loader, jpeg4py_loader_w_failsafe
from .lasot import Lasot
from .got10k import Got10k
from .tracking_net import TrackingNet
from .imagenetvid import ImagenetVID
from .coco import MSCOCO
from .tao import TAO
from .coco_seq import MSCOCOSeq
from .youtubevos import YouTubeVOS
from .davis import Davis
from .lvis import LVIS
from .ecssd import ECSSD
from .msra10k import MSRA10k
from .hku_is import HKUIS
from .sbd import SBD
from .tao import TAO
from .crowdhuman import CrowdHumanSeq
from .mot import MOT
from .lvis_seq import LVISSeq
from .synthetic_video import SyntheticVideo
from .synthetic_video_blend import SyntheticVideoBlend


def get_trainsets(dataset_names, data_dir, image_loader=opencv_loader):
  assert isinstance(dataset_names, list)
  trainsets = []
  data_dir = Path(data_dir)
  for name in dataset_names:
    name = name.lower()
    if name == "lasot":
      trainsets.append(
          Lasot(data_dir / 'LaSOTBenchmark',
                split='train',
                image_loader=image_loader))
    elif name == "got10k_vottrain":
      trainsets.append(
          Got10k(data_dir / 'GOT10k' / 'train',
                 split='vottrain',
                 image_loader=image_loader))
    elif name == "got10k_train":
      trainsets.append(
          Got10k(data_dir / 'GOT10k' / 'train',
                 split='train',
                 image_loader=image_loader))
    elif name == "got10k_votval":
      trainsets.append(
          Got10k(data_dir / 'GOT10k' / 'train',
                 split='votval',
                 image_loader=image_loader))
    elif name == "youtubevos":
      trainsets.append(
          YouTubeVOS(data_dir / 'YoutubeVOS',
                 split='train',
                 multiobj=False,
                 image_loader=image_loader))
    elif name == "davis":
      trainsets.append(
          Davis(data_dir / 'Davis',
                 split='train',
                 multiobj=False,
                 image_loader=image_loader))
    elif name == "tao":
      trainsets.append(
          TAO(data_dir / 'TAO', split='train', image_loader=image_loader))
    elif name == "mot":
      trainsets.append(
          MOT(data_dir / 'MOT', split='train', image_loader=image_loader))
    elif name == "mots":
      trainsets.append(
          MOT(data_dir / 'MOT', split='train', image_loader=image_loader, version='MOTS'))
    elif name == "mot_trainhalf":
      trainsets.append(
          MOT(data_dir / 'MOT', split='train_half', image_loader=image_loader))
    elif name == "coco17":
      trainsets.append(
          MSCOCOSeq(data_dir / 'coco',
                    version="2017",
                    image_loader=image_loader))
    elif name == "lvis":
      trainsets.append(
          LVISSeq(data_dir / 'coco', split='train', image_loader=image_loader))
    elif name == "crowdhuman":
      trainsets.append(
          CrowdHumanSeq(data_dir / 'CrowdHuman', image_loader=image_loader))
    elif name == "vid":
      trainsets.append(ImagenetVID(data_dir / 'VID', image_loader=image_loader))
    elif name == "trackingnet":
      # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
      trainsets.append(
          TrackingNet(data_dir / 'TrackingNet',
                      image_loader=image_loader, set_ids=range(12)))
    else:
      raise NotImplementedError(f'{name} not supported')

  return trainsets