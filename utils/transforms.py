import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomResnetTransforms:
    """ Class to define image transformations for training """

    def train_transforms(means, stds):
        """
        Image Augmentation transformations to apply on the training images


        Args:
            means: mean of all the images in the training dataset
            stds: standard deviation of all the images in the training dataset
        
        Returns:
            transforms: the list of all transformation to be applied on training images
        """
        return A.Compose(
            [
                A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                #A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=means),
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )

    def test_transforms(means, stds):
        """
        Image Augmentation transformations to apply on the testing images


        Args:
            means: mean of all the images in the testing dataset
            stds: standard deviation of all the images in the testing dataset
        
        Returns:
            transforms: the list of all transformation to be applied on testing images
        """
        return A.Compose(
            [
                A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )