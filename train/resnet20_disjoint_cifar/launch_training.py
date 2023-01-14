#%%
import submitit
from dataclasses import asdict
from config import Config
from trainCifar5 import train
import numpy as np
#%%
executor = submitit.AutoExecutor(folder="log_train")
executor.update_parameters(timeout_min=100, slurm_partition="debug", slurm_gres="gpu:1")
executor.update_parameters(slurm_array_parallelism=5)
configs = []
for i in range(4):
    class_idxs = np.arange(10)
    np.random.shuffle(class_idxs)
    model1_classes = class_idxs[:5]
    model2_classes = class_idxs[5:]
    configs.append(asdict(Config("/srv/share/jbjorner3/checkpoints/REPAIR/", f"resnet20x4_CIFAR5_clses{model1_classes}", f"resnet20x4_CIFAR5_clses{model2_classes}", model1_classes, model2_classes)))
jobs = executor.map_array(train, configs)
for job in jobs:
    print(job.result())
#%%
# if __name__ == "__main__":
    # split1 = [0, 1, 2, 3, 4]
    # split2 = [5, 6, 7, 8, 9]
    # config = Config("/srv/share/jbjorner3/checkpoints/REPAIR/", f"resnet20x4_CIFAR5_clses{split1}", f"resnet20x4_CIFAR5_clses{split2}", split1, split2)
    # config = asdict(config)
    # job = executor.submit(train, config)
    # print(job.result())
    # train(config)

# %%
