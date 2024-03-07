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
class DataDividerConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    params_train_size: float
    params_test_size: float
    params_random_state: int

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: str
    training_data: str
    all_params: dict
    mlflow_uri: str
