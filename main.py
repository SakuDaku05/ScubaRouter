"""
Entry point for manual testing during the prep days, before kickoff.

Usage:
    python main.py "What is the capital of France?"

With use_mock: true in config/models.yaml (the default), this runs the
full pipeline end-to-end using canned mock responses -- no real model
servers required yet. Flip use_mock to false once your local server and
Fireworks credentials are ready.
"""
import sys

from src.config import load_config
from src.local_model import LocalModel
from src.remote_model import RemoteModel
from src.pipeline import RoutingPipeline


def build_pipeline() -> RoutingPipeline:
    config = load_config()
    use_mock = config["routing"].get("use_mock", True)

    local_config = {**config["local"], "use_mock": use_mock}
    remote_config = {**config["remote"], "use_mock": use_mock}

    local_model = LocalModel(local_config)
    remote_model = RemoteModel(remote_config)
    return RoutingPipeline(local_model, remote_model, config)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    pipeline = build_pipeline()
    result = pipeline.handle_query(query, task_type="qa")
    print(result)
