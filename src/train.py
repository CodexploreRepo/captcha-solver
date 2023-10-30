from sched import scheduler
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
from model import CaptchaModel
import engine
from pprint import pprint

def run_training():
    image_files =[file_path for file_path in config.DATA_DIR.glob("*.png")] 
    #../../../728n8.png => .split("/")[-1] = 728n8.png => 728n8
    targets_orig = [str(x).split("/")[-1][:-4] for x in image_files]
    #targets = [['p', '5', 'g', '5', 'm'], ['e', '7', '2', 'c', 'd'], ...] 
    targets = [[c for c in x] for x in targets_orig]
    #flatten the target
    targets_flat = [c for c_list in targets for c in c_list]
    print(f"Unique Target Vocab: {np.unique(targets_flat)}")
    
    #Encoding the target
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1 # since we want 0 for unknowns
    
    # print(targets_enc)
    print(len(lbl_enc.classes_))
    
    train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets =\
    model_selection.train_test_split(image_files, targets_enc, targets_orig, test_size=0.2, random_state=2022)

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )
    test_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = False
    )
    
    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        valid_captcha_preds = []
        
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_captcha_preds.extend(current_preds)
        
        combined = list(zip(test_orig_targets, valid_captcha_preds))
        pprint(combined[:10])
        # test_dup_rem = [remove_duplicates(c) for c in test_orig_targets]
        # accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={valid_loss}"
        )
        scheduler.step(valid_loss)

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2) #Change back to (bs, timestamp, prediction)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1 #need to -1 for all the predicted class
            if k == -1: #unknown characters
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds
    
if __name__ == "__main__":
    run_training()
    # 75 values: "°6666°°°dddddd°77°8°h°°°°" where ° is unknown character
    