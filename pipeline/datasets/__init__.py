# Import all dataset loaders so they register themselves
from pipeline.datasets.base import BaseDataset, register_dataset, get_dataset, DATASET_REGISTRY
from pipeline.datasets.hotpotqa import HotpotQADataset
from pipeline.datasets.docvqa import DocVQADataset
from pipeline.datasets.gqa import GQADataset
from pipeline.datasets.altumint import AltumintDataset