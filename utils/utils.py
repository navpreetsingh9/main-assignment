import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from typing import Union
import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def imshow(img):
    """
    Display image


    Args:
        img: Given image to display
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def denormalize_image(image, mean, std):
    """
    Denormalize an image given its mean and standard deviation.

    Parameters:
        image (torch.Tensor): The input image tensor to denormalize.
        mean (tuple or list): The mean used for normalization.
        std (tuple or list): The standard deviation used for normalization.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    # Convert the mean and std to tensors
    mean_tensor = torch.tensor(mean).reshape(1, -1, 1, 1)
    std_tensor = torch.tensor(std).reshape(1, -1, 1, 1)

    # Denormalize the image
    denormalized_image = image * std_tensor + mean_tensor

    return denormalized_image

def show_images(img_loader, class_map, count=10):
    """
    Display multiple images
    

    Args:
        img_loader (Dataloader): dataloader for training data
        classes (int, optional): Number of images to show. Defaults to 10.
    """
    # Print Random Samples
    if not count % 10 == 0:
        return

    classes = list(class_map.keys())
    fig = plt.figure(figsize=(15, 5))
    for imgs, labels in img_loader:
        for i in range(count):
            ax = fig.add_subplot(int(count / 10), 10, i + 1, xticks=[], yticks=[])
            ax.set_title(f"{classes[labels[i]]}")
            plt.imshow(np.clip(imgs[i], 0, 1).cpu().numpy().transpose(1, 2, 0).astype(float))
        break
    plt.show()

    
def print_summary(model, input_size=(1, 28, 28)):
    """Print Model summary

    Args:
        model (Net): Model Instance
        input_size (tuple, optional): Input size. Defaults to (1, 28, 28).
    """
    return summary(model, input_size=input_size)

def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)

def load_weights_from_path(model, path):
    """load weights from file

    Args:
        model (Net): Model instance
        path (str): Path to weights file

    Returns:
        Net: loaded model
    """
    model.load_state_dict(torch.load(path))
    return model

def get_incorrect_predictions(model, loader, device):
    """Get all incorrect predictions

    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu

    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect

def get_all_predictions(model, loader, device):
    """Get All predictions for model

    Args:
        model (Net): Trained Model 
        loader (Dataloader): instance of dataloader
        device (str): Which device to use cuda/cpu

    Returns:
        tuple: all predicted values and their targets
    """
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in loader:
            data, targets = data.to(device), target.to(device)
            all_targets = torch.cat(
                (all_targets, targets),
                dim=0
            )
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            )

    return all_preds, all_targets


def prepare_confusion_matrix(all_preds, all_targets, class_map):
    """Prepare Confusion matrix

    Args:
        all_preds (list): List of all predictions
        all_targets (list): List of all actule labels
        class_map (dict): Class names

    Returns:
        tensor: confusion matrix for size number of classes * number of classes
    """
    stacked = torch.stack((
        all_targets, all_preds
    ),
        dim=1
    ).type(torch.int64)

    no_classes = len(class_map)

    # Create temp confusion matrix
    confusion_matrix = torch.zeros(no_classes, no_classes, dtype=torch.int64)

    # Fill up confusion matrix with actual values
    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    return confusion_matrix

def plot_incorrect_predictions(predictions, class_map, count=10):
    """Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 10 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.keys())

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Correct_Label/Incorrect_Prediction", fontsize=16, fontweight="bold")
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/10), 10, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        plt.imshow(np.clip(d, 0, 1).cpu().numpy().transpose(1, 2, 0))
        if i+1 == count:
            break
    plt.show()
    

def plot_network_performance(epochs, train_losses, test_losses, train_acc, test_acc):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), train_losses, 'g', label='Training loss')
    plt.plot(range(epochs), test_losses, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), train_acc, 'g', label='Training Accuracy')
    plt.plot(range(epochs), test_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_gradcam(model, incorrect, class_map, use_cuda, mean, std, count=10):
    target_layers  = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    if not count % 10 == 0:
        return

    classes = list(class_map.keys())
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Correct_Label/Incorrect_Prediction", fontsize=16, fontweight="bold")
    for i, (imgs, labels, pred, output) in enumerate(incorrect):
        input_tensor = imgs.unsqueeze(0)
        rgb_img = denormalize_image(input_tensor, mean, std).transpose(3, 1).numpy()[0]
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        ax = fig.add_subplot(int(20 / 10), 10, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[labels.item()]}/{classes[pred.item()]}')
        plt.imshow(visualization)
        if i+1 == count:
            break
    plt.show()