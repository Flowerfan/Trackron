from trackron.structures import SequenceList
from trackron.data.datasets.trainsets.davis import Davis
from .otb import OTBDataset
from .got10k import GOT10KDataset
from .mot import MOTDataset
from .tao import TAODataset
from .lasot import LaSOTDataset
from .trackingnet import TrackingNetDataset
from .uav import UAVDataset
from .nfs import NFSDataset


def load_dataset(data_root, name: str, version, split):
  """ Import and load a single dataset."""
  name = name.lower()
  if name == 'otb':
    dataset = OTBDataset(data_root / 'OTB')
  elif name == 'nfs':
    dataset = NFSDataset(data_root / 'NFS')
  elif name == 'uav':
    dataset = UAVDataset(data_root / 'UAV')
  elif name == 'lasot':
    dataset = LaSOTDataset(data_root / 'LaSOTBenchmark')
  elif name == 'got10k':
    dataset = GOT10KDataset(data_root / 'GOT10k', split=split)
  elif name == 'trackingnet':
    dataset = TrackingNetDataset(data_root / 'TrackingNet')
  elif name == 'mot':
    dataset = MOTDataset(data_root / 'MOT', version=version, split=split)
  elif name == 'mot_pub':
    # assert version == '17'
    dataset = MOTDataset(data_root / 'MOT', version=version, split=split, public_detection=True)
  elif name == 'tao':
    dataset = TAODataset(data_root / 'TAO', split=split)
  elif name == 'davis':
    dataset = Davis(data_root / 'Davis', split=split)
  else:
    raise NotImplementedError(f'not support {name}')
  return dataset.get_sequence_list()


def get_dataset(data_root, data_names, versions, splits):
  """ Get a single or set of datasets."""
  dset = SequenceList()
  for idx, name in enumerate(data_names):
    dset.extend(load_dataset(data_root, name, versions[idx], splits[idx]))
  return dset


def build_test_dataset(dataset):
  pass