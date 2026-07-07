import sys
sys.path.insert(0, '.')

import yaml
cfg = yaml.safe_load(open('config/models.yaml'))
cfg['routing']['use_mock'] = True
cfg['routing']['use_supra'] = False  # skip model load in smoke test

from src.local_model import LocalModel
from src.remote_model import RemoteModel
from src.pipeline import RoutingPipeline

local  = LocalModel({**cfg['local'],  'use_mock': True})
remote = RemoteModel({**cfg['remote'], 'use_mock': True})
pipeline = RoutingPipeline(local, remote, cfg)

tests = [
    'What is the capital of France?',
    'What is 17 * 24?',
    'Write a Python function to reverse a string.',
    'Summarize: The cat sat on the mat.',
    'Extract all persons from: Barack Obama met with Angela Merkel.',
    'Classify the sentiment: I love this product!',
    'If all A are B and all B are C, are all A C?',
]

for query in tests:
    r = pipeline.handle_query(query)
    tt = r['task_type']
    route = r['route']
    conf = r['confidence']
    print(f"type={tt:20s}  route={route:8s}  conf={conf:.2f}  | {query[:50]}")
