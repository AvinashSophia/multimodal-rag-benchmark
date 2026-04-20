# Import all dataset loaders so they register themselves
from pipeline.datasets.base import BaseDataset, register_dataset, get_dataset, DATASET_REGISTRY
from pipeline.datasets.altumint_aws import AltumintAWSDataset
from pipeline.datasets.hotpotqa_aws import HotpotQAAWSDataset
from pipeline.datasets.docvqa_aws import DocVQAAWSDataset
from pipeline.datasets.gqa_aws import GQAAWSDataset
