from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: str
    unzip_dir: Path
    data_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    directory: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_dir: Path

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: str
    training_data: str
    all_params: dict
    mlflow_uri: str
