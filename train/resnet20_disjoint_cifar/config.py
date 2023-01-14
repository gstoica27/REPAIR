"""
File for creating configs for running training scripts
Specify save location, account details, and and training hyperparameters.
Helps with replication and finding models.
"""
#%%
from dataclasses import dataclass
from typing import List
@dataclass
class Config:
    """
    checkpoint_save_dir: where to save the model after training
    model1_name: the name where model_1 will be saved
    model2_name: the name where model_2 will be saved
    model1_classes: an array of the classes to be trained on model 1
    model2_classes: an array of the classes to be trained on model 2
    """
    checkpoint_save_dir: str
    model1_name: str
    model2_name: str
    model1_classes: List[int]
    model2_classes: List[int]


if __name__ == "__main__":
    cfg = (Config(
        "bill", 
        "1", 
        "2", 
        [1,2,3], 
        [4,5,6])
    )
    print(cfg)
# resnet20x4_CIFAR5_clses{model1_classes.tolist()}


# %%
