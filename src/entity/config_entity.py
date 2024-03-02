from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    data_path: Path

class ModelNameConfig:
    """Model Configurations"""
    model_name: str = "lightgbm"
    fine_tuning: bool = False
