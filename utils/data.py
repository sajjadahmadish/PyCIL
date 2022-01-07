import os
import glob
import h5py
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


# ===============================================================================

# ModelNet40

# ===============================================================================


def fetch_modelnet40(root):
    if not os.path.exists(root):
        www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], root))
        os.system('rm %s' % zipfile)


def load_data(root, partition, download=True):
    if download:
        fetch_modelnet40(root)
    all_data = []
    all_label = []
    g = sorted(glob.glob(os.path.join(root, 'ply_data_%s*.h5' % partition)))
    for h5_file in g:
        f = h5py.File(h5_file)
        label = f['label'][:].astype('int64')
        temp_data = f['data'][:].astype('float32')
        f.close()
        all_data.append(temp_data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class TranslatePointcloud(object):

    def __init__(self, num_points=1024, train=True):
        self.num_points = num_points
        self.train = train

    def __call__(self, sample):
        if self.train:
            xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
            xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
            x = sample[: self.num_points]
            translated_pointcloud = np.add(np.multiply(x, xyz1), xyz2).astype('float32')
            np.random.shuffle(translated_pointcloud)
            return translated_pointcloud.transpose(1, 0)
        else:
            return sample[: self.num_points].transpose(1, 0)


class iModelNet40(iData):
    train_trsf = [TranslatePointcloud(1024)]
    test_trsf = [TranslatePointcloud(1024, False)]
    class_order = [2, 3, 4, 10, 14, 17, 19, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 39, 5, 16, 23, 25, 37, 9,
                   12, 13, 20, 24, 0, 1, 6, 34, 38, 7, 8, 11, 15, 18]
    data = None
    targets = None
    label_to_id = None
    id_to_label = None
    train_data = None
    train_targets = None
    test_data = None
    test_targets = None

    def _create_class_mapping(self, path):
        label_to_id = {}
        self.class_order = []
        with open(os.path.join(path, "shape_names.txt")) as f:
            for i, line in enumerate(f):
                ls = line.strip().split()
                label_to_id[ls[0]] = i
                self.class_order.append(i)  # Classes are already in the right order.

        id_to_label = {v: k for k, v in label_to_id.items()}
        return label_to_id, id_to_label

    def base_dataset(self, root, train=True, download=False):
        directory = os.path.join(root, 'modelnet40_ply_hdf5_2048')
        self.data, self.targets = load_data(directory, partition='train' if train else 'test', download=download)

        self.label_to_id, self.id_to_label = self._create_class_mapping(directory)

        return self

    def download_data(self):
        data_path = './data'

        train_dataset = self.base_dataset(data_path, train=True, download=True)
        test_dataset = self.base_dataset(data_path, train=False, download=True)

        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)