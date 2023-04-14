from torch.utils.data import Dataset, DataLoader
import numpy as np
class SpectroDataset(Dataset):
    def __init__(self, databulk, labelbulk, transform=None):
        # self.paths = paths
        # self.class_map = { "animal": 1, "cloth" : 2, "device" : 3, "food" : 4, "holiday" : 5, "insect" : 6, "instrument" : 7, "organ" : 8, "person" : 9, "place" : 10, "thing" : 11, "tool" : 12, "weapon" : 13}
        # self.class_map = {  'animal':0, 'person':1, 'tool': 2}
        self.class_map = {  'noun1':0, 'prespeech':1}
        self.databulk = databulk
        self.labelbulk = labelbulk
        self.transform = transform
        # self.class_map = {  'animate':0, 'inanimate':1}
    def __getitem__(self,idx):

        data = self.databulk[idx]
        label = int(self.labelbulk[idx])
        # label = int(label)
        
        # data = (data*255).astype(np.uint8)
        data = data.astype(np.float32)
        data = data.reshape(53,50)
        # data = data.astype(np.uint8)
        # # img = Image.fromarray(data, 'RGB')
        # im1 = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        # # image = np.dstack([data*255]*3)
        # image = im1
        # image = np.dstack([data]*3)
        # print(data.shape)
        image = data

        if self.transform:
          image = self.transform(image)
        # image = im1
        # transform = transforms.Compose([
        #     # transforms.RandomHorizontalFlip(0.5),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),           
        #     # transforms.Resize((64, 64))
        #     # transforms.RandomResizedCrop(32)
        # ])

        # img_tensor = transform(image)
        
        # return img_tensor, label
        return image, label
    
    def __len__(self):
        return len(self.databulk)