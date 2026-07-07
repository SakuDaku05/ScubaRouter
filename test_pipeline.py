import json
import os
import time

from src.config import load_config
from src.local_model import LocalModel
from src.remote_model import RemoteModel
from src.pipeline import RoutingPipeline
from src.logger import Logger

def main():
    # Load config
    config = load_config("config/models.yaml")

    # Force remote model to be mock so we don't need Fireworks/Groq API keys for this test
    # (We only care about testing the local routing and LM Studio right now)
    config["remote"]["use_mock"] = True
    config["local"]["use_mock"] = False

    local_model = LocalModel(config["local"])
    remote_model = RemoteModel(config["remote"])
    
    # Turn off Supra for this test just to see the pure Local Model + Verifier logic
    # (Or leave it on, but since we don't have Supra downloaded, it might error. Let's leave it as is, Supra is in the config.)
    
    pipeline = RoutingPipeline(local_model, remote_model, config, logger=Logger())

    with open("eval/benchmark_dataset.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)[:5]  # Just the first 5 for a quick test

    print("="*70)
    print(f"Testing {len(tasks)} tasks against local LM Studio + Mock Remote")
    print("="*70)
    
    results = []
    total_remote_tokens = 0
    t0 = time.time()

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['task_type'].upper()} ({task['difficulty']})")
        print(f"Q: {task['prompt']}")
        
        task_start = time.time()
        record = pipeline.handle_query(
            query=task["prompt"],
            task_type=task["task_type"],
            difficulty=task["difficulty"]
        )
        elapsed = time.time() - task_start
        
        route = record["route"]
        tokens = record["remote_tokens_used"]
        total_remote_tokens += tokens
        conf = record["confidence"]
        
        print(f"Route: {route.upper()}  |  Conf: {conf:.2f}  |  Tokens: {tokens}  |  Time: {elapsed:.1f}s")
        if route == "local":
            print(f"Local Answer: {record['local_answer'][:150]}...")
        else:
            print(f"Remote Answer: {record['final_answer'][:150]}...")
            
        results.append(record)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total tasks: {len(tasks)}")
    print(f"Local handled: {sum(1 for r in results if r['route'] == 'local')} (0 scored tokens)")
    print(f"Escalated:     {sum(1 for r in results if r['route'] == 'escalate')} (paid tokens)")
    print(f"Total remote tokens used: {total_remote_tokens}")
    print(f"Total time taken: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
