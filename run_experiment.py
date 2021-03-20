import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np



from logger import Logger



device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')




# Prepare dataset
batch_size = 128
train_mean = [107.59252, 103.2752, 106.84143]
train_std = [63.439133, 59.521027, 63.240288]
# Preprocessing
transform = T.Compose([
                T.Normalize(train_mean, train_std)
            ])

#train_set = TensorDataset(torch.load('train_x.pt'), torch.load('train_y.pt'))
#val_set = TensorDataset(torch.load('val_x.pt'), torch.load('val_y.pt'))
#test_set = TensorDataset(torch.load('test_x.pt'), torch.load('test_y.pt'))

train_rotate_set = TensorDataset(torch.load('train_x_rotate.pt'), torch.load('train_y_rotate.pt'))
val_rotate_set = TensorDataset(torch.load('val_x_rotate.pt'), torch.load('val_y_rotate.pt'))
test_rotate_set = TensorDataset(torch.load('test_x_rotate.pt'), torch.load('test_y_rotate.pt'))

#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

train_rotate_loader = DataLoader(train_rotate_set, batch_size=batch_size, shuffle=True)
val_rotate_loader = DataLoader(val_rotate_set, batch_size=batch_size, shuffle=True)
test_rotate_loader = DataLoader(test_rotate_set, batch_size=batch_size, shuffle=True)



# Set up training pipelines
def train_main(model, optimizer, loader_train, loader_val, epochs=1, model_path=None, early_stop_patience = 0):
    """
    Train the main branch
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Logger object with loss and accuracy data
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    logger = Logger()
    best_loss = float('inf')
    for e in range(epochs):
        num_correct = 0
        num_samples = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\r[Epoch {e}, Batch {t}] train_loss: {loss.item()}", end='')

        # Conclude Epoch
        train_loss = loss.item()
        train_acc = float(num_correct) / num_samples
        val_loss, val_acc = evaluate_main(model, loader_val)
        logger.log(train_loss, train_acc, val_loss, val_acc)
        
        # Early Stopping
        if logger.check_early_stop(early_stop_patience):
            print("[Early Stopped]")
            break
        else:
            if best_loss > val_loss:
                print(f"\r[Epoch {e}] train_acc: {train_acc}, val_acc:{val_acc}, val_loss improved from %.4f to %.4f. Saving model to {model_path}." % (best_loss, val_loss))
                if model_path is not None:
                    torch.save(model.state_dict(), model_path)
                best_loss = val_loss
            else:
                print(f"\r[Epoch {e}] train_acc: {train_acc}, val_acc:{val_acc}, val_loss did not improve from %.4f" % (best_loss))

        

def train_both(model, optimizer, loader_train, loader_val, epochs=1, model_path=None, early_stop_patience = 0):
    """
    Train the main and auxillary branch
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Logger object with loss and accuracy data
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    logger = Logger()
    best_loss = float('inf')
    for e in range(epochs):
        num_correct = 0
        num_samples = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss_main = F.cross_entropy(scores[0], y[:, 0])
            loss_auxillary = F.cross_entropy(scores[1], y[:, 1])
            loss = loss_main + loss_auxillary
            # loss = loss_main

            _, preds = scores[0].max(1)
            num_correct += (preds == y[:, 0]).sum()
            num_samples += preds.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\r[Epoch {e}, Batch {t}] train_loss: {loss.item()}", end='')

        # Conclude Epoch
        train_loss = loss.item()
        train_acc = float(num_correct) / num_samples
        val_loss, val_acc = evaluate_both(model, loader_val)
        logger.log(train_loss, train_acc, val_loss, val_acc)
        
        # Early Stopping
        if logger.check_early_stop(early_stop_patience):
            print("[Early Stopped]")
            break
        else:
            if best_loss > val_loss:
                print(f"\r[Epoch {e}] train_acc: {train_acc}, val_acc:{val_acc}, val_loss improved from %.4f to %.4f. Saving model to {model_path}." % (best_loss, val_loss))
                if model_path is not None:
                    torch.save(model.state_dict(), model_path)
                best_loss = val_loss
            else:
                print(f"\r[Epoch {e}] train_acc: {train_acc}, val_acc:{val_acc}, val_loss did not improve from %.4f" % (best_loss))





def evaluate_main(model, loader):
    """
    Evaluate main branch accuracy
    Outputs: loss and accuracy
    """
    num_correct = 0
    num_samples = 0
    ave_loss = 0.0
    count = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            # print(scores.shape)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            ave_loss += loss.item()
            count += 1
        acc = float(num_correct) / num_samples
        return ave_loss / count, acc


def evaluate_both(model, loader):
    """
    Evaluate main branch accuracy in model with two predictions
    Outputs: loss and accuracy
    """
    num_correct = 0
    num_samples = 0
    ave_loss = 0.0
    count = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss_main = F.cross_entropy(scores[0], y[:, 0])
            # print(scores.shape)
            _, preds = scores[0].max(1)
            num_correct += (preds == y[:, 0]).sum()
            # print(f"num_correct: {num_correct}")
            num_samples += preds.size(0)
            # print(f"num_samples: {num_samples}")
            ave_loss += loss_main.item()
            count += 1
        acc = float(num_correct) / num_samples
        return ave_loss / count, acc

def evaluate_debug(model, loader):
    num_correct = 0
    num_samples = 0
    for t, (x, y) in enumerate(loader):
        model.train()  # put model to training mode
        x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)

        scores = model(x)
        loss_main = F.cross_entropy(scores[0], y[:, 0])
        loss_auxillary = F.cross_entropy(scores[1], y[:, 1])
        loss = loss_main + loss_auxillary

        _, preds = scores[0].max(1)
        num_correct += (preds == y[:, 0]).sum()
        num_samples += preds.size(0)

    # Conclude Epoch
    # train_loss = loss.item()
    train_acc = float(num_correct) / num_samples
    return 0.0, train_acc



def ttt_online(model, loader, loader_spinned, optimizer):
    """
    Online TTT with image spinning task
    Outputs: loss and accuracy
    """
    for x, y in loader_spinned:
        x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        loss_auxillary = F.cross_entropy(scores[1], y[:, 1])
        optimizer.zero_grad()
        loss_auxillary.backward()
        optimizer.step()
    return evaluate_main(model, loader)


'''
from models import ResNetMainBranch

# Experiment 1: Train a baseline ResNet18: no branch
lr = 1e-3
wd = 1e-5
from models import ResNetMainBranch, ResNetTwoBranch
model = ResNetTwoBranch()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
train_main(model, optimizer, train_loader, val_loader, epochs=50, model_path='model.pth', early_stop_patience=5)

model_path = 'model.pth'
model = ResNetTwoBranch()
params = torch.load(model_path)
model.load_state_dict(params)
model = model.to(device=device)
loss, acc = evaluate_main(model, test_loader)
print(loss, acc)
'''


# Experiment 2: Train a ResNet18 with auxillary branch
from models import ResNetTwoBranch

lr = 1e-3
wd = 1e-5
from models import ResNetTwoBranch
model = ResNetTwoBranch()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
train_both(model, optimizer, train_rotate_loader, train_rotate_loader, epochs=50, model_path='exp_2_model.pth', early_stop_patience=5)


model_path = 'exp_2_model.pth'
model = ResNetTwoBranch()
params = torch.load(model_path)
model.load_state_dict(params)
model = model.to(device=device)
loss, acc = evaluate_both(model, test_rotate_loader)
print(loss, acc)

