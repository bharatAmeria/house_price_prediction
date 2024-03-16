from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

class ModelNameConfig:
    """Model Configurations"""

    model_name: str = "lightgbm"
    fine_tuning: bool = False
