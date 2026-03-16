from torchvision import datasets, transforms as T
from torch.utils.data import Dataset


MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

class SimCLRAugmentations:
    def __init__(self, size=224):
        self.size = size

        color_jitter = T.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2
        )

        self.transform = T.Compose([
            T.RandomResizedCrop(self.size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __call__(self, image):
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2



class SimCLRDataset(Dataset):
    def __init__(self, dataset_name, size=32):
        self.size = size
        self.transform = SimCLRAugmentations(size=size)

        dataset_class = self.choose_dataset(dataset_name)

        self.dataset = dataset_class(
            root="data",
            train=True,
            download=True
        )

    def choose_dataset(self, dataset_name):
        if dataset_name == "cifar100":
            return datasets.CIFAR100
        elif dataset_name == "cifar10":
            return datasets.CIFAR10
        else:
            raise ValueError("Unsupported dataset")

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        view1, view2 = self.transform(image)

        return view1, view2

    def __len__(self):
        return len(self.dataset)


