import wandb
import time
import numpy as np
import sys

PROJECT="debugging"
ENTITY="tim-hays"
MODEL_NAME="test_model"
RUN_TIME_SECONDS = 15
EVAL_STEPS = 1
ARRAY_SIZE = 2**10

settings = wandb.Settings()
run = wandb.init(
    project=PROJECT,
    entity=ENTITY,
    job_type="train",
    settings=settings,
    config={
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 100
    }
)

print(f"Starting run with args {sys.argv}")
start_time = time.time()

i = 0
time_elapsed = 0
while time_elapsed < RUN_TIME_SECONDS:
    i += 1
    arr1 = np.random.rand(ARRAY_SIZE, ARRAY_SIZE)
    arr2 = np.random.rand(ARRAY_SIZE, ARRAY_SIZE)
    prod = np.matmul(arr1, arr2)
    avg = np.average(prod)
    time_elapsed = time.time() - start_time
    if i % 10 == 0:
        wandb.log({
            "steps": i,
            "avg": avg,
            "time_elapsed": time_elapsed
        })

run.log_code()
print("Run completed")
wandb.finish()