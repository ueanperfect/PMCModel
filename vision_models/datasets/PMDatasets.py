import os

from torch.utils.data.dataset import Dataset
import cv2


class PMDatasets(Dataset):
    def __init__(self, data_path, data_type, transforms=None):
        super(PMDatasets, self).__init__()
        self.data_path = data_path
        assert data_type == 'train' or data_type == 'test' or data_type == 'val'
        self.data_path = os.path.join(self.data_path, data_type)
        classes_list = {'dry': 0, 'ice': 1, 'wet': 2}
        self.image_path_list = []
        self.id_list = []
        for class_item in classes_list:
            classes_path = os.path.join(self.data_path,class_item)
            self.image_path_list = self.image_path_list + [os.path.join(classes_path,name) for name in os.listdir(os.path.join(self.data_path, class_item))]
            self.id_list = self.id_list + [classes_list[class_item] for i in
                                           range(len(os.listdir(os.path.join(self.data_path, class_item))))]
        self.transforms = transforms

    def __getitem__(self, index):
        img = cv2.imread(self.image_path_list[index])
        label = self.id_list[index]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.image_path_list)