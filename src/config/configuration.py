from src.entity.config_entity import DataIngestionConfig, EvaluationConfig
from src.utils.common import read_yaml, create_directories
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            data_path=config.data_path
        )

        return data_ingestion_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/gurgaon_properties/gurgaon_properties.csv",
            mlflow_uri=" ",
            all_params=self.params,
        )
        return eval_config


