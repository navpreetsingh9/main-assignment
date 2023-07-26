import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse

from models.resnet import ResNet18, ResNet34
from utils.dataset import means, stds, class_map, Cifar10SearchDataset
from utils.transforms import CustomResnetTransforms
from utils.utils import show_images, print_summary, get_incorrect_predictions, denormalize_image
from utils.utils import plot_incorrect_predictions, plot_network_performance, show_gradcam
from utils.train import train, test, get_sgd_optimizer, get_adam_optimizer, get_onecyclelr_scheduler, get_lr_finder
from utils.train import train_losses, test_losses, train_acc, test_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay factor')
    parser.add_argument('--use_scheduler', action='store_true', help='Use scheduler')
    parser.add_argument('--end_lr', default=10, type=float, help='end_lr for one cycle policy')
    parser.add_argument('--seed', default=1, type=int, help='Seed')
    parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle images')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch Size')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin_memory')
    parser.add_argument('--show_images', action='store_true', help='Show images')
    parser.add_argument('--img_count', default=10, type=int, help='Count of images to show')
    parser.add_argument('--show_misclassified', action='store_true', help='Show misclassified images')
    parser.add_argument('--show_gradcam', action='store_true', help='Show gradcam of images')
    parser.add_argument('--print_summary', action='store_false', help='Print model summary')
    parser.add_argument('--model', default="resnet18", type=str, help='Which model to use for training')
    parser.add_argument('--optimizer', default="adam", type=str, help='Which optimizer to use for training')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs for training')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda = torch.cuda.is_available()
    print("CUDA Available?", torch.cuda.is_available())
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # Data
    print('==> Preparing data..')
    train_transforms = CustomResnetTransforms.train_transforms(means, stds)
    test_transforms = CustomResnetTransforms.test_transforms(means, stds)

    train_dataset = Cifar10SearchDataset(root='./data', train=True,
                                            download=True, transform=train_transforms)
    test_dataset = Cifar10SearchDataset(root='./data', train=False,
                                        download=True, transform=test_transforms)

    dataloader_args = dict(shuffle=args.shuffle, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.pin_memory) if cuda else dict(shuffle=args.shuffle, batch_size=args.batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    if args.show_images:
        show_images(train_loader, class_map, count=args.img_count)

    # Model
    print('==> Building model..')
    if args.model == "resnet18":
        model = ResNet18().to(device)
    else: #default
        model = ResNet34().to(device) 

    if args.print_summary:
        print_summary(model, (1, 3, 36, 36))

    print('==> Initializing optimizer and scheduler..')
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = get_sgd_optimizer(model, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = get_adam_optimizer(model, args.lr, weight_decay=args.weight_decay)
    else: #default
        optimizer = get_adam_optimizer(model, args.lr, weight_decay=args.weight_decay)

    if args.use_scheduler:
        max_lr = get_lr_finder(model, optimizer, criterion, device, train_loader, args.end_lr)
        scheduler = get_onecyclelr_scheduler(optimizer, max_lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    print('==> Training the model..')
    for epoch in range(1,args.epochs+1):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
        if args.use_scheduler:
            scheduler.step()
    
    print('==> Finished Training..')
    plot_network_performance(args.epochs, train_losses, test_losses, train_acc, test_acc)
    
    if args.show_misclassified:
        incorrect = get_incorrect_predictions(model, test_loader, device)
        plot_incorrect_predictions(incorrect, class_map, args.img_count)
    
    if args.show_gradcam:
        show_gradcam(model, test_loader, class_map, cuda, means, stds, args.img_count)

    