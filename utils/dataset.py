from torchvision import datasets

means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]
class_map = {
    "PLANE": 0,
    "CAR": 1,
    "BIRD": 2,
    "CAT": 3,
    "DEER": 4,
    "DOG": 5,
    "FROG": 6,
    "HORSE": 7,
    "SHIP": 8,
    "TRUCK": 9,
}


class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):
        """
        Initialise Cifar10SearchDataset


        Args:
            root: path where the dataset will be stored
            train: Boolean to indicate whether the dataset will be used for training or not
            download: Boolean to indicate whether to download the dataset
            transform: List of transformations applied on the dataset
        """
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        """
        Get image and its corresponding label for the given index

        Args:
            index: index of the image/class to retrieve
        
        Returns:
            image: image at corresponding index
            label: label at corresponding index
        """
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label