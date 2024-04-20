import yaml


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)


def load_config(file_path):
    with open(file_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            return Config(**config_dict)
        except yaml.YAMLError as exc:
            print(exc)
