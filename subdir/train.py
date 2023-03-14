import wandb
import time
import numpy as np

PROJECT="debugging"
ENTITY="tim-hays"
MODEL_NAME="test_model"
RUN_TIME_SECONDS = 15
EVAL_STEPS = 1
ARRAY_SIZE = 2**10
GIB = 1024 * 1024 * 1024

#settings = wandb.Settings(disable_git=True, disable_job_creation=True)
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

print("Starting run")

def main():
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

    # if random.random() > 0.5:
    #     raise Exception

# if random.random() > 0.5:
main()

# inc = lambda path: True
# run.log_code(include_fn=inc)
# wandb.save("requirements.frozen.txt")
# art = wandb.Artifact('job-source-debugging-train.py', type='job')
# art.add_file('./requirements.frozen.txt', 'requirements.frozen.txt')
# art.add_file('./wandb-job.json', 'wandb-job.json')
# wandb.log_artifact(art)

run.log_code()
print("Run completed")
wandb.finish()