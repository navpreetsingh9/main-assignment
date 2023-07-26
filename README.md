# Main Repo for ERA

This repository contains the main PyTorch implementation of image classification using CNNs. It includes a training script (`main.py`) to train any model on any dataset.

## Requirements

- Python 3.x
- PyTorch (>=1.0)
- torchvision
- numpy
- torch-summary
- torch_lr_finder
- grad-cam

## Usage

To run the training script, use the following command:

```
python main.py [--options]
```

### Options

- `--lr` (default=0.01): Learning Rate.
- `--momentum` (default=0.9): Momentum factor for the optimizer.
- `--weight_decay` (default=5e-4): Weight decay factor for L2 regularization.
- `--use_scheduler`: Use the One Cycle Learning Rate Scheduler (default=False).
- `--end_lr` (default=10): The upper bound of the learning rate range for the One Cycle Policy.
- `--seed` (default=1): Random seed for reproducibility.
- `--shuffle` (default=True): Shuffle the images in the dataset (True/False).
- `--batch_size` (default=512): Batch Size for training.
- `--num_workers` (default=4): Number of workers for data loading.
- `--pin_memory` (default=True): Use pinned memory for faster data transfer to GPU (True/False).
- `--show_images`: Show sample images from the training dataset.
- `--img_count` (default=10): Number of images to display if `--show_images` is enabled.
- `--show_misclassified`: Show misclassified images after training.
- `--show_gradcam`: Show GradCAM visualization for some images after training.
- `--print_summary` (default=True): Print the model summary.
- `--model` (default="resnet18"): Choose the model for training ("resnet18" or "resnet34").
- `--optimizer` (default="adam"): Choose the optimizer for training ("sgd" or "adam").
- `--epochs` (default=20): Number of epochs for training.

### Example Usage

To train a ResNet18 model with the Adam optimizer and One Cycle Learning Rate Scheduler:

```
python main.py --model resnet18 --optimizer adam --use_scheduler
```

## Data Preparation

The dataset will be automatically downloaded and prepared during the first run of the script. The data will be saved in the `./data` directory.

## Results

During training, the script will display the training and validation accuracy and loss for each epoch. After training, a plot displaying the training and validation losses and accuracies over epochs will be shown.

## Visualization

### Sample Images

You can use the `--show_images` option to visualize some sample images from the training dataset.

### Misclassified Images

If `--show_misclassified` is enabled, the script will display some misclassified images after training.

### GradCAM Visualization

If `--show_gradcam` is enabled, GradCAM visualizations will be shown for some images after training.

## Model Summary

You can print the model summary using the `--print_summary` option. Note that this will show the summary for a randomly initialized model, not the final trained model.

## Utils

This repository includes the following utility modules:

- `dataset.py`: Contains utility functions and classes for loading and preprocessing the CIFAR-10 dataset.
- `train.py`: Includes training and evaluation functions.
- `transforms.py`: Defines custom data transformations used during training and testing.
- `utils.py`: Contains various utility functions used in the training script.

## Models

The `models` folder contains the following model implementations:

- `resnet.py`: Includes the definitions of ResNet18 and ResNet34 models used for image classification.

## License

This project is licensed under the [MIT License](https://chat.openai.com/LICENSE).

Feel free to modify and use this code for your image classification tasks. If you find it useful, please consider giving it a star on GitHub!