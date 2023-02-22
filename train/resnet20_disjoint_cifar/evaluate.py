#%%
import torch
from torch.cuda.amp import autocast
import numpy as np
import re
import os
from resnet20 import resnet20
from tqdm import tqdm
import torchvision
import torchvision.transforms as T
import sys
# turn the relative path ./ViT_pytorch into an absolute path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ViT_pytorch')))
# /nethome/jbjorner3/dev/REPAIR/train/resnet20_disjoint_cifar/ViT_pytorch
from models.modeling import VisionTransformer, CONFIGS
from vitutils.data_utils import get_cifar50_loader

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#%%
DEVICE = "cuda"
from clip import clip
test_dset_classes = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_dset_classes]).to(DEVICE)
model_clip, preprocess = clip.load('ViT-B/32')
with torch.no_grad():
    text_features = model_clip.encode_text(text_inputs).to(DEVICE)
text_features.shape
text_features /= text_features.norm(dim=-1, keepdim=True)


def evaluate(model, loader, class_vectors, remap_class_idxs=None, return_confusion=False, DEVICE='cuda', cifar_num=5):
    model.eval()
    correct = 0
    total = 0
    confusion = np.zeros((cifar_num*2, cifar_num*2))
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            encodings = model(inputs.to(DEVICE))
            encodings = encodings[0] if isinstance(encodings, tuple) else encodings # .to(list(dict(model.state_dict()).items())[0][1].dtype)
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
            outputs = normed_encodings @ class_vectors.T
            pred = outputs.argmax(dim=1)
            if remap_class_idxs is not None:
                correct += (remap_class_idxs[labels].to(DEVICE) == pred).sum().item()
            else:
                correct += (labels.to(DEVICE) == pred).sum().item()
            confusion[labels.cpu().numpy(), pred.cpu().numpy()] += 1
            total += inputs.shape[0]
    if return_confusion:
        return correct / total, confusion / confusion.sum(-1, keepdims=True)
    else:
        return correct / total


def evaluate_cifar(model_save_name, DEVICE='cuda'):
    # add ability to load vit models like the path cifar50-MAE-optim-0-lr-0.001-[50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]_checkpoint.bin
    #%%
    file_name = model_save_name.split("/")[-1]
    classes_string = re.search("\[([ \n\d]+)\]", file_name).group(1)
    model_classes = re.findall("\d+", classes_string)
    model_classes = np.array([int(num) for num in model_classes])
    #%%
    cifar_num = re.search("CIFAR(\d+)", file_name, re.IGNORECASE).group(1)
    cifar_num = int(cifar_num) # the cifar number is listed in halves.
    class_idxs = np.zeros(cifar_num *2, dtype=int)
    class_idxs[model_classes] = np.arange(cifar_num)
    class_idxs = torch.from_numpy(class_idxs)

    
    if "resnet" in file_name:
        _, test_dset, _, test_loader = get_data_loaders_resnet(cifar_num, model_classes)
        model = get_resnet_model(model_save_name, DEVICE)
    elif "MAE" in file_name or "Google" in file_name:
        _, _, test_dset, test_loader = get_data_loaders_vit(model_classes)
        model = get_vit_model(model_save_name, DEVICE)

    
    class_vecs = text_features[model_classes].to(DEVICE)

    return evaluate(model, test_loader, class_vecs, class_idxs, DEVICE=DEVICE, cifar_num=cifar_num)
def evaluate_cifar_vit(model, test_loader, model_classes, DEVICE='cuda'):
    # implicit assumption that the vit model is 50 classes and is one of the trained variants MAE or Google
    cifar_num = 50
    class_idxs = np.zeros(cifar_num *2, dtype=int)
    class_idxs[model_classes] = np.arange(cifar_num)
    class_idxs = torch.from_numpy(class_idxs)

    class_vecs = text_features[model_classes].to(DEVICE)
    
    return evaluate(model, test_loader, class_vecs, class_idxs, DEVICE=DEVICE, cifar_num=cifar_num)


def get_resnet_model(model_save_name, DEVICE):
    model = resnet20(w=4).to(DEVICE)
    sd = torch.load(model_save_name, map_location=torch.device(DEVICE))
    model.load_state_dict(sd)
    return model
def get_vit_model(model_save_name, DEVICE):
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, 224, zero_head=True, num_classes=512)
    model.load_state_dict(torch.load(model_save_name, map_location=torch.device(DEVICE)))
    return model.to(DEVICE)

def get_data_loaders_vit(model_classes):
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.dataset = 'cifar50'
    args.local_rank = -1
    args.img_size = 224
    args.eval_batch_size = 64
    args.train_batch_size = 512
    args.model_split = model_classes
    train_dset, train_loader, test_dset, test_loader = get_cifar50_loader(args, return_dsets=True)
    return train_dset, train_loader, test_dset, test_loader
def get_data_loaders_resnet(cifar_num, model_classes):
    """ returns the dataloaders for train and test depending on the split and cifar number """
    
    if cifar_num == 50:
        CIFAR_MEAN = [129.3105, 124.1085, 112.404]
        CIFAR_STD = [68.2125, 65.4075, 70.4055]
    else:
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]

    normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)
    denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), 255/np.array(CIFAR_STD))

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    if cifar_num == 50:
        train_dset = torchvision.datasets.CIFAR100(root='/tmp', train=True,
                                                download=True, transform=train_transform)
        test_dset = torchvision.datasets.CIFAR100(root='/tmp', train=False,
                                            download=True, transform=test_transform)
    else:
        train_dset = torchvision.datasets.CIFAR10(root='/tmp', train=True,
                                                download=True, transform=train_transform)
        test_dset = torchvision.datasets.CIFAR10(root='/tmp', train=False,
                                            download=True, transform=test_transform)
    valid_examples = [i for i, (_, label) in tqdm(enumerate(train_dset)) if label in model_classes]

    train_aug_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dset, valid_examples), batch_size=500, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=500, shuffle=False, num_workers=8)
    train_aug_loader = torch.utils.data.DataLoader(train_dset, batch_size=500, shuffle=True, num_workers=8)

    test_valid_examples = [i for i, (_, label) in tqdm(enumerate(test_dset)) if label in model_classes]

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dset, test_valid_examples), batch_size=500, shuffle=False, num_workers=8
    )
    return train_dset, test_dset, train_aug_loader, test_loader


#%%
if __name__ == "__main__":
    # model_save_name = "/srv/share/jbjorner3/checkpoints/REPAIR/resnet20x4_CIFAR50_clses[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49].pth.tar"
    # model_save_name = "/srv/share/jbjorner3/checkpoints/REPAIR/resnet20x4_CIFAR5_clses[5 0 4 7 9].pth.tar"
    #%%
    # checkpoints_path = '/srv/share/jbjorner3/checkpoints/REPAIR'
    checkpoints_path = "/nethome/jbjorner3/dev/REPAIR/train/resnet20_disjoint_cifar/ViT_pytorch/output"
    model_classes = np.arange(50, 100, dtype=int)
    _, _, _, test_loader = get_data_loaders_vit(model_classes)
    for file in os.listdir(checkpoints_path):
        print(file)
        # print(evaluate_cifar(os.path.join(checkpoints_path, file)))
        model = get_vit_model(os.path.join(checkpoints_path, file), DEVICE)
        print(evaluate_cifar_vit(model, test_loader, model_classes, DEVICE))

    
