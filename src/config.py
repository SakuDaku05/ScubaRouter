import yaml
from dotenv import load_dotenv


def load_config(path: str = "config/models.yaml") -> dict:
    load_dotenv()
    with open(path) as f:
        return yaml.safe_load(f)
