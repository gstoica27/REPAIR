# %%
import os
import sys
import pdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
import torchvision
import torchvision.transforms as T

from sys import platform
from collections import defaultdict

import wandb
#%%
def train(config):
    checkpoint_save_dir = config["checkpoint_save_dir"]
    model1_classes = config["model1_classes"]
    model2_classes = config["model2_classes"]
    model1_name = config["model1_name"]
    model2_name = config["model2_name"]
    DEVICE = 'mps' if platform == 'darwin' else 'cuda'

    # %%
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
    train_dset = torchvision.datasets.CIFAR10(root='/tmp', train=True,
                                            download=True, transform=train_transform)
    test_dset = torchvision.datasets.CIFAR10(root='/tmp', train=False,
                                            download=True, transform=test_transform)

    # class_idxs = np.arange(10)
    # np.random.shuffle(class_idxs)
    # model1_classes = class_idxs[:5]
    # model2_classes = class_idxs[5:]

    # model1_classes= np.array([3, 2, 0, 6, 4])
    # model2_classes = np.array([5, 7, 9, 8, 1])

    valid_examples1 = [i for i, (_, label) in tqdm(enumerate(train_dset)) if label in model1_classes]
    valid_examples2 = [i for i, (_, label) in tqdm(enumerate(train_dset)) if label in model2_classes]

    assert len(set(valid_examples1).intersection(set(valid_examples2))) == 0, 'sets should be disjoint'

    train_aug_loader1 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dset, valid_examples1), batch_size=500, shuffle=True, num_workers=8
    )
    train_aug_loader2 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dset, valid_examples2), batch_size=500, shuffle=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=500, shuffle=False, num_workers=8)

    # %%
    train_aug_loader = torch.utils.data.DataLoader(train_dset, batch_size=500, shuffle=True, num_workers=8)

    # %%
    test_valid_examples1 = [i for i, (_, label) in tqdm(enumerate(test_dset)) if label in model1_classes]
    test_valid_examples2 = [i for i, (_, label) in tqdm(enumerate(test_dset)) if label in model2_classes]

    # %%
    test_loader1 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dset, test_valid_examples1), batch_size=500, shuffle=False, num_workers=8
    )
    test_loader2 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dset, test_valid_examples2), batch_size=500, shuffle=False, num_workers=8
    )

    # %%
    class_idxs = np.zeros(10, dtype=int)
    class_idxs[model1_classes] = np.arange(5)
    class_idxs[model2_classes] = np.arange(5)
    class_idxs = torch.from_numpy(class_idxs)
    class_idxs

    # %%
    def save_model(model, i):
        sd = model.state_dict()
        path = os.path.join(
            # '/Users/georgestoica/Downloads',
            checkpoint_save_dir,
            '%s.pth.tar' % i
        )
        torch.save(model.state_dict(), path)

    def load_model(model, i):
        path = os.path.join(
            # '/Users/georgestoica/Downloads',
            checkpoint_save_dir,
            '%s.pth.tar' % i
        )
        sd = torch.load(path, map_location=torch.device(DEVICE))
        model.load_state_dict(sd)

    # %%
    # given two networks net0, net1 which each output a feature map of shape NxCxWxH
    # this will reshape both outputs to (N*W*H)xC
    # and then compute a CxC correlation matrix between the outputs of the two networks
    def run_corr_matrix(net0, net1, epochs=1, norm=True, loader=train_aug_loader):
        n = epochs*len(loader)
        mean0 = mean1 = std0 = std1 = None
        with torch.no_grad():
            net0.eval()
            net1.eval()
            for _ in range(epochs):
                for i, (images, _) in enumerate(tqdm(loader)):
                    img_t = images.float().to(DEVICE)
                    out0 = net0(img_t)
                    out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                    out0 = out0.reshape(-1, out0.shape[2]).double()

                    out1 = net1(img_t)
                    out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                    out1 = out1.reshape(-1, out1.shape[2]).double()

                    mean0_b = out0.mean(dim=0)
                    mean1_b = out1.mean(dim=0)
                    std0_b = out0.std(dim=0)
                    std1_b = out1.std(dim=0)
                    outer_b = (out0.T @ out1) / out0.shape[0]

                    if i == 0:
                        mean0 = torch.zeros_like(mean0_b)
                        mean1 = torch.zeros_like(mean1_b)
                        std0 = torch.zeros_like(std0_b)
                        std1 = torch.zeros_like(std1_b)
                        outer = torch.zeros_like(outer_b)
                    mean0 += mean0_b / n
                    mean1 += mean1_b / n
                    std0 += std0_b / n
                    std1 += std1_b / n
                    outer += outer_b / n
                    
        cov = outer - torch.outer(mean0, mean1)
        if cov.isnan().sum() > 0: pdb.set_trace()
        if norm:
            corr = cov / (torch.outer(std0, std1) + 1e-4)
            return corr.to(torch.float32)
        else:
            return cov.to(torch.float32)

    # modifies the weight matrices of a convolution and batchnorm
    # layer given a permutation of the output channels
    def permute_output(perm_map, conv, bn):
        pre_weights = [
            conv.weight,
            bn.weight,
            bn.bias,
            bn.running_mean,
            bn.running_var,
        ]
        for i, w in enumerate(pre_weights):
            if len(pre_weights) == i + 1:
                w @ perm_map.t()
            if len(w.shape) == 4:
                transform = torch.einsum('ab,bcde->acde', perm_map, w)
            elif len(w.shape) == 2:
                transform = perm_map @ w
            else:
                transform = w @ perm_map.t()
            # assert torch.allclose(w[perm_map.argmax(-1)], transform)
            w.data = transform
            # w.data = w[perm_map]
                

    # modifies the weight matrix of a convolution layer for a given
    # permutation of the input channels
    def permute_input(perm_map, after_convs):
        if not isinstance(after_convs, list):
            after_convs = [after_convs]
        post_weights = [c.weight for c in after_convs]
        for w in post_weights:
            if len(w.shape) == 4:
                transform = torch.einsum('abcd,be->aecd', w, perm_map.t())
            elif len(w.shape) == 2:
                transform = w @ perm_map.t()
        #     assert torch.allclose(w[:, perm_map.argmax(-1)], transform)
            w.data = transform
    #         w.data = w[:, perm_map, :, :]

    def permute_cls_output(perm_map, linear):
        for w in [linear.weight, linear.bias]:
            w.data = perm_map @ w

    # %%
    def transform_model(
        model0, 
        model1, 
        model_to_alter, 
        transform_fn, 
        prune_threshold=-torch.inf, 
        module2io=defaultdict(lambda: dict())
    ):
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                return x

        perm_map, collapse_totals = transform_fn(Subnet(model0), Subnet(model1))
        permute_output(perm_map, model_to_alter.conv1, model_to_alter.bn1)
        permute_output(perm_map, model_to_alter.layer1[0].conv2, model_to_alter.layer1[0].bn2)
        permute_output(perm_map, model_to_alter.layer1[1].conv2, model_to_alter.layer1[1].bn2)
        permute_output(perm_map, model_to_alter.layer1[2].conv2, model_to_alter.layer1[2].bn2)
        permute_input(perm_map, 
                    [
                        model_to_alter.layer1[0].conv1, 
                        model_to_alter.layer1[1].conv1, 
                        model_to_alter.layer1[2].conv1
                    ]
                    )
        permute_input(perm_map, [model_to_alter.layer2[0].conv1, model_to_alter.layer2[0].shortcut[0]])
        
        module2io['conv1']['output'] = collapse_totals
        module2io['bn1']['output'] = collapse_totals
        module2io['layer1.0.conv2']['output'] = collapse_totals
        module2io['layer1.0.bn2']['output'] = collapse_totals
        module2io['layer1.1.conv2']['output'] = collapse_totals
        module2io['layer1.1.bn2']['output'] = collapse_totals
        module2io['layer1.2.conv2']['output'] = collapse_totals
        module2io['layer1.2.bn2']['output'] = collapse_totals

        module2io['layer1.0.conv1']['input'] = collapse_totals
        module2io['layer1.1.conv1']['input'] = collapse_totals
        module2io['layer1.2.conv1']['input'] = collapse_totals
        module2io['layer2.0.conv1']['input'] = collapse_totals
        module2io['layer2.0.shortcut.0']['input'] = collapse_totals
        
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                return x

        perm_map, collapse_totals = transform_fn(Subnet(model0), Subnet(model1))
        permute_output(perm_map, model_to_alter.layer2[0].conv2, model_to_alter.layer2[0].bn2)
        permute_output(perm_map, model_to_alter.layer2[0].shortcut[0], model_to_alter.layer2[0].shortcut[1])
        permute_output(perm_map, model_to_alter.layer2[1].conv2, model_to_alter.layer2[1].bn2)
        permute_output(perm_map, model_to_alter.layer2[2].conv2, model_to_alter.layer2[2].bn2)

        permute_input(perm_map, [model_to_alter.layer2[1].conv1, model_to_alter.layer2[2].conv1])
        permute_input(perm_map, [model_to_alter.layer3[0].conv1, model_to_alter.layer3[0].shortcut[0]])
        
        module2io['layer2.0.conv2']['output'] = collapse_totals
        module2io['layer2.0.bn2']['output'] = collapse_totals
        module2io['layer2.0.shortcut.0']['output'] = collapse_totals
        module2io['layer2.0.shortcut.1']['output'] = collapse_totals
        module2io['layer2.1.conv2']['output'] = collapse_totals
        module2io['layer2.1.bn2']['output'] = collapse_totals
        module2io['layer2.2.conv2']['output'] = collapse_totals
        module2io['layer2.2.bn2']['output'] = collapse_totals

        module2io['layer2.1.conv1']['input'] = collapse_totals
        module2io['layer2.2.conv1']['input'] = collapse_totals
        module2io['layer3.0.conv1']['input'] = collapse_totals
        module2io['layer3.0.shortcut.0']['input'] = collapse_totals
        
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                return x

        perm_map, collapse_totals = transform_fn(Subnet(model0), Subnet(model1))
        permute_output(perm_map, model_to_alter.layer3[0].conv2, model_to_alter.layer3[0].bn2)
        permute_output(perm_map, model_to_alter.layer3[0].shortcut[0], model_to_alter.layer3[0].shortcut[1])
        permute_output(perm_map, model_to_alter.layer3[1].conv2, model_to_alter.layer3[1].bn2)
        permute_output(perm_map, model_to_alter.layer3[2].conv2, model_to_alter.layer3[2].bn2)

        permute_input(perm_map, [model_to_alter.layer3[1].conv1, model_to_alter.layer3[2].conv1])
        model_to_alter.linear.weight.data = model_to_alter.linear.weight @ perm_map.t()
        
        module2io['layer3.0.conv2']['output'] = collapse_totals
        module2io['layer3.0.bn2']['output'] = collapse_totals
        module2io['layer3.0.shortcut.0']['output'] = collapse_totals
        module2io['layer3.0.shortcut.1']['output'] = collapse_totals
        module2io['layer3.1.conv2']['output'] = collapse_totals
        module2io['layer3.1.bn2']['output'] = collapse_totals
        module2io['layer3.2.conv2']['output'] = collapse_totals
        module2io['layer3.2.bn2']['output'] = collapse_totals

        module2io['layer3.1.conv1']['input'] = collapse_totals
        module2io['layer3.2.conv1']['input'] = collapse_totals
        module2io['linear']['input'] = collapse_totals
        
        class Subnet(nn.Module):
            def __init__(self, model, nb=9):
                super().__init__()
                self.model = model
                self.blocks = []
                self.blocks += list(model.layer1)
                self.blocks += list(model.layer2)
                self.blocks += list(model.layer3)
                self.blocks = nn.Sequential(*self.blocks)
                self.bn1 = model.bn1
                self.conv1 = model.conv1
                self.linear = model.linear
                self.nb = nb

            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.blocks[:self.nb](x)
                block = self.blocks[self.nb]
                x = block.conv1(x)
                x = block.bn1(x)
                x = F.relu(x)
                return x

        blocks1 = []
        blocks1 += list(model_to_alter.layer1)
        blocks1 += list(model_to_alter.layer2)
        blocks1 += list(model_to_alter.layer3)
        blocks1 = nn.Sequential(*blocks1)
        
        block_idx2name = {
            0: 'layer1.0',
            1: 'layer1.1',
            2: 'layer1.2',
            3: 'layer2.0',
            4: 'layer2.1',
            5: 'layer2.2',
            6: 'layer3.0',
            7: 'layer3.1',
            8: 'layer3.2'
        }
        for nb, (block_idx, layer_name) in zip(range(9), block_idx2name.items()):
            perm_map, collapse_totals = transform_fn(Subnet(model0, nb=nb), Subnet(model1, nb=nb))
            block = blocks1[nb]
            permute_output(perm_map, block.conv1, block.bn1)
            permute_input(perm_map, [block.conv2])

            module2io[layer_name + '.conv1']['output'] = collapse_totals
            module2io[layer_name + '.bn1']['output'] = collapse_totals
            module2io[layer_name + '.conv2']['output'] = collapse_totals
        
        return model_to_alter, module2io


    # %%
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.init as init

    def _weights_init(m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)

    class LambdaLayer(nn.Module):
        def __init__(self, lambd):
            super(LambdaLayer, self).__init__()
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
    #             self.shortcut = LambdaLayer(lambda x:
    #                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(planes)
                )


        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, w=1, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = w*16

            self.conv1 = nn.Conv2d(3, w*16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(w*16)
            self.layer1 = self._make_layer(block, w*16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, w*32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, w*64, num_blocks[2], stride=2)
            self.linear = nn.Linear(w*64, 512)

            self.apply(_weights_init)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion

            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


    def resnet20(w=1):
        return ResNet(BasicBlock, [3, 3, 3], w=w)

    # %%
    def train(save_key, model, train_loader, test_loader, class_vectors, remap_class_idxs):
        optimizer = SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4)
        # optimizer = Adam(model.parameters(), lr=0.05)
        
        # Adam seems to perform worse than SGD for training ResNets on CIFAR-10.
        # To make Adam work, we find that we need a very high learning rate: 0.05 (50x the default)
        # At this LR, Adam gives 1.0-1.5% worse accuracy than SGD.
        
        # It is not yet clear whether the increased interpolation barrier for Adam-trained networks
        # is simply due to the increased test loss of said networks relative to those trained with SGD.
        # We include the option of using Adam in this notebook to explore this question.

        EPOCHS = 100
        ne_iters = len(train_loader)
        lr_schedule = np.interp(np.arange(1+EPOCHS*ne_iters), [0, 5*ne_iters, EPOCHS*ne_iters], [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

        scaler = GradScaler()
        loss_fn = CrossEntropyLoss()
        
        losses = []
        for _ in tqdm(range(EPOCHS)):
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    encodings = model(inputs.to(DEVICE))
                    normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                    logits = (100.0 * normed_encodings @ class_vectors.T)
                    remapped_labels = remap_class_idxs[labels].to(DEVICE)
                    loss = loss_fn(logits, remapped_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(loss.item())
        print(evaluate(
            model, test_loader, 
            class_vectors=class_vectors, 
            remap_class_idxs=remap_class_idxs
        ))
        save_model(model, save_key)

    # %%
    # evaluates accuracy
    def evaluate(model, loader, class_vectors, remap_class_idxs=None, return_confusion=False):
        model.eval()
        correct = 0
        total = 0
        confusion = np.zeros((10, 10))
        with torch.no_grad(), autocast():
            for inputs, labels in loader:
                encodings = model(inputs.to(DEVICE))
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
    # evaluates loss
    def evaluate1(model, loader, class_vectors, remap_class_idxs):
        model.eval()
        losses = []
        pdb.set_trace()
        with torch.no_grad(), autocast():
            for inputs, labels in loader:
                encodings = model(inputs.to(DEVICE))
                normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                outputs = normed_encodings @ class_vectors.T
                loss = F.cross_entropy(outputs, remap_class_idxs[labels].to(DEVICE))
                losses.append(loss.item())
        return np.array(losses).mean()

    # %%
    from clip import clip

    # %%
    test_dset.classes

    # %%
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_dset.classes]).to(DEVICE)

    # %%
    model, preprocess = clip.load('ViT-B/32', DEVICE)

    # %%
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # %%
    text_features.shape

    # %%
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # %%
    class_vecs1 = text_features[model1_classes]
    class_vecs2 = text_features[model2_classes]
    # class_vecs1 /= class_vecs1.norm(dim=-1, keepdim=True)
    # class_vecs2 /= class_vecs2.norm(dim=-1, keepdim=True)
    #%%
    def train_single_model(single_config):
        model_name, model_classes, class_vecs, train_aug_loader, test_loader = single_config
        single_config_dict = {
            "model_name": model_name, 
            "model_classes": model_classes, 
            "class_vecs": class_vecs, 
            "train_aug_loader": train_aug_loader, 
            "test_loader": test_loader
        }
        if not os.path.exists(
            os.path.join(
                checkpoint_save_dir,
                f'{model_name}.pth.tar'
            )
        ):
            print('training model...')
            # wandb.init(project="ModelFusion-cifar5", config=single_config_dict)
            model = resnet20(w=4).to(DEVICE)
            # wandb.watch(model, log_freq=100)
            train(
                f'{model_name}', 
                model=model, 
                class_vectors=class_vecs,
                train_loader=train_aug_loader,
                test_loader=test_loader,
                remap_class_idxs=class_idxs
            )
    # %%
    for single_config in \
        [(model1_name, model1_classes, class_vecs1, train_aug_loader1, test_loader1),
        (model2_name, model2_classes, class_vecs2, train_aug_loader2, test_loader2)]:
        train_single_model(single_config)

        
    # if not os.path.exists(
    #     os.path.join(
    #         checkpoint_save_dir,
    #         f'{model2_name}.pth.tar'
    #     )
    # ):
    #     print('training model...')
    #     model2 = resnet20(w=4).to(DEVICE)
    #     train(
    #         f'{model2_name}', 
    #         model=model2, 
    #         class_vectors=class_vecs2,
    #         train_loader=train_aug_loader2,
    #         test_loader=test_loader2,
    #         remap_class_idxs=class_idxs
    #     )

