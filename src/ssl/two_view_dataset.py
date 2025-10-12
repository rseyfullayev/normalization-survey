class TwoViewWrapper:

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return (x1, x2), label