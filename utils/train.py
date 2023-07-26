from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder

train_losses = []
test_losses = []
train_acc = []
test_acc = []
lrs = []

def get_lr(optimizer):
    """
    Tracking learning rate during training


    Args:
        optimizer: optimizer used for training
    
    Returns:
        lr: the learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def get_sgd_optimizer(model, lr, momentum=0.9, weight_decay=5e-4):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_adam_optimizer(model, lr, weight_decay=1e-4):
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_lr_finder(model, optimizer, criterion, device, train_loader, end_lr):
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=200, step_mode="exp")
    _, suggested_lr = lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return suggested_lr


def get_onecyclelr_scheduler(optimizer, max_lr, steps_per_epoch, epochs):
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=5/epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    train_loss = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)

        loss = criterion(y_pred, target)
        train_loss += loss.item()
        lrs.append(get_lr(optimizer))

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} LR={get_lr(optimizer)} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        
    train_losses.append(train_loss/processed)
    train_acc.append(100*correct/processed)
        

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))