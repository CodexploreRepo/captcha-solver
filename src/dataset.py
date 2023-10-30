import albumentations
import torch
import numpy as np

from PIL import Image, ImageFile
from sklearn import preprocessing
import config

#In case, there is a truncated image
#it should be loaded as well
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        """_summary_

        Args:
            image_paths (_type_): _description_
            targets (_type_): _description_
            resize (w, h): _description_. Defaults to None.
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.aug = albumentations.Compose([
            albumentations.Normalize(always_apply=True)
        ])
    
    def __getitem__(self, idx):
        #read the image
        image = Image.open(self.image_paths[idx]).convert("RGB") #4 channels to 3 channels RGB
        target = self.targets[idx]
        
        if self.resize is not None:
            #PIL resize function: get (w=300, h=75)
            image = image.resize((self.resize[0], self.resize[1]), resample=Image.Resampling.BILINEAR)
            #print(image.size) #(300, 75)
        
        image = np.array(image)
        #print(image.shape) #(75, 300, 3) = (h, w, channel)
        
        augmented = self.aug(image=image)
        image = augmented["image"]
        #transpose from (h, w, channel) to (channel, h, w) format
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.long)
        }
        
    
    def __len__(self):
        return (len(self.image_paths))
    

if __name__ == "__main__":
    image_files =[file_path for file_path in config.DATA_DIR.glob("*.png")]
    targets_orig = [str(x).split("/")[-1][:-4] for x in image_files]
    
    #targets = [['p', '5', 'g', '5', 'm'], ['e', '7', '2', 'c', 'd'], ...] 
    targets = [[c for c in x] for x in targets_orig]
    #flatten the target
    targets_flat = [c for c_list in targets for c in c_list]
    
    #Encoding the target
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1 # since we want 0 for unknowns
    
    ds = ClassificationDataset(image_files, targets=targets_enc, resize=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    print(ds[0]['image'].numpy().shape)