# from apex import amp
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize, HorizontalFlip, RandomBrightnessContrast, \
    Normalize
from albumentations.pytorch import ToTensor
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import IntracranialDataset
from models import ResNeXtModel

saved_model_dir = '../input/resnext32x8dcheckpoint/'

dir_csv = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'
test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test/'
train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'
train_metadata_csv = '../input/rsna-intracranial-sequence-metadata/train_metadata_noidx.csv'
test_metadata_csv = '../input/rsna-intracranial-sequence-metadata/test_metadata_noidx.csv'

n_classes = 6
n_epochs = 3
batch_size = 32

COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

# Read train and test data
train = pd.read_csv(os.path.join(dir_csv, 'stage_2_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_2_sample_submission.csv'))

# Read metadata for train/validation split
test_metadata_noidx = pd.read_csv(test_metadata_csv)
train_metadata_noidx = pd.read_csv(train_metadata_csv)

# Prepare train table
train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']

# Remove invalid PNGs
png = glob.glob(os.path.join(train_images_dir, '*.dcm'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)

train = train[train['Image'].isin(png)]

merged_train = pd.merge(left=train, right=train_metadata_noidx, how='left', left_on='Image', right_on='ImageId')

train_series = train_metadata_noidx['SeriesInstanceUID'].unique()
valid_series = train_series[21000:]
train_series = train_series[:21000]

print(len(train_series))
print(len(valid_series))

train_df = merged_train[merged_train['SeriesInstanceUID'].isin(train_series)]
valid_df = merged_train[merged_train['SeriesInstanceUID'].isin(valid_series)]

print(len(train_df))
print(len(valid_df))

train_df.to_csv('train.csv', index=False)
print(train_df['any'].value_counts())
valid_df.to_csv('valid.csv', index=False)
print(valid_df['any'].value_counts())

# Prepare test table
test[['ID', 'Image', 'Diagnosis']] = test['ID'].str.split('_', expand=True)
test['Image'] = 'ID_' + test['Image']
test = test[['Image', 'Label']]
test.drop_duplicates(inplace=True)

test.to_csv('test.csv', index=False)

# Data loaders
transform_train = Compose([Resize(256, 256),
                           Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111], max_pixel_value=1.),
                           HorizontalFlip(),
                           ShiftScaleRotate(),
                           RandomBrightnessContrast(),
                           ToTensor()])

transform_test = Compose([Resize(256, 256),
                          Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111], max_pixel_value=1.),
                          ToTensor()])

transform_tta = Compose([Resize(256, 256),
                         HorizontalFlip(),
                         ShiftScaleRotate(),
                         Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111], max_pixel_value=1.),
                         ToTensor()])

train_dataset = IntracranialDataset(
    csv_file='train.csv', path=train_images_dir, transform=transform_train, labels=True)

valid_dataset = IntracranialDataset(
    csv_file='valid.csv', path=train_images_dir, transform=transform_train, labels=True)

test_dataset = IntracranialDataset(
    csv_file='test.csv', path=test_images_dir, transform=transform_test, labels=False)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

print(len(data_loader_train))
print(len(data_loader_valid))
print(len(data_loader_test))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNeXtModel()

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    print('-' * 10)

    model.train()
    tr_loss = 0

    for step, batch in enumerate(data_loader_train):
        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        if step % 512 == 0:
            epoch_loss = tr_loss / (step + 1)
            print('Training Loss at {}: {:.4f}'.format(step, epoch_loss))

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))
    print('-----------------------')

    model.eval()
    tr_loss = 0

    auc_preds = []
    auc_truths = []

    for step, batch in enumerate(data_loader_valid):
        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)

        tr_loss += loss.item()

        auc_preds.append(outputs.view(-1, 6).detach().cpu().numpy())
        auc_truths.append(labels.view(-1, 6).detach().cpu().numpy())

    epoch_loss = tr_loss / len(data_loader_valid)
    print('Validation Loss: {:.4f}'.format(epoch_loss))

    roc_preds = np.concatenate(auc_preds)

    roc_truths = np.concatenate(auc_truths)

    for tp in range(0, 6):
        print(COLS[tp], roc_auc_score(roc_truths[:, tp], roc_preds[:, tp]), )
    print('-----------------------')

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    # 'amp': amp.state_dict()
}
torch.save(checkpoint, 'model.pt')

# Save embeddings/predictions
PCA_BATCHES = 1000
model.eval()

train_embed_dict = {}

for i, x_batch in enumerate(tqdm(data_loader_train)):
    x_images = x_batch['image_id']
    x_batch = x_batch["image"]
    x_batch = x_batch.to(device, dtype=torch.float)

    if i > PCA_BATCHES:
        break

    with torch.no_grad():
        _, embed = model(x_batch)

        for x, y in zip(x_images, embed):
            e = y.squeeze().detach().cpu().numpy()
            train_embed_dict[x] = e

emb_stat = np.array(list(train_embed_dict.values()))
print(np.mean(emb_stat), np.std(emb_stat))  # 0.4707987 0.7724904

# pca = PCA()
# pca.fit(emb_stat)
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)')
# plt.title('Explained Variance')
# plt.show()

pca = PCA(n_components=120)

pca.fit(emb_stat)

model.eval()

# TRAIN
train_pred_dict = {}
train_embed_dict = {}

for i, x_batch in enumerate(tqdm(data_loader_train)):
    x_images = x_batch['image_id']
    x_batch = x_batch["image"]
    x_batch = x_batch.to(device, dtype=torch.float)

    with torch.no_grad():

        pred, embed = model(x_batch)
        pred = torch.sigmoid(pred)

        for x, y in zip(x_images, pred):
            train_pred_dict[x] = y.detach().cpu().numpy()
        for x, y in zip(x_images, embed):
            e = y.squeeze().detach().cpu().numpy()
            e = np.expand_dims(e, axis=0)
            train_embed_dict[x] = pca.transform(e)[0]

train_embed_df = pd.DataFrame.from_dict(train_embed_dict, orient='index')
train_embed_df.to_csv('train_embeds.csv')

train_pred_df = pd.DataFrame.from_dict(train_pred_dict, orient='index')
train_pred_df.to_csv('train_preds.csv')

# VALID
valid_pred_dict = {}
valid_embed_dict = {}

for i, x_batch in enumerate(tqdm(data_loader_valid)):
    x_images = x_batch['image_id']
    x_batch = x_batch["image"]
    x_batch = x_batch.to(device, dtype=torch.float)

    with torch.no_grad():

        pred, embed = model(x_batch)
        pred = torch.sigmoid(pred)

        for x, y in zip(x_images, pred):
            valid_pred_dict[x] = y.detach().cpu().numpy()
        for x, y in zip(x_images, embed):
            e = y.squeeze().detach().cpu().numpy()
            e = np.expand_dims(e, axis=0)
            valid_embed_dict[x] = pca.transform(e)[0]

valid_embed_df = pd.DataFrame.from_dict(valid_embed_dict, orient='index')
valid_embed_df.to_csv('valid_embeds.csv')

valid_pred_df = pd.DataFrame.from_dict(valid_pred_dict, orient='index')
valid_pred_df.to_csv('valid_preds.csv')

# TEST
test_pred_dict = {}
test_embed_dict = {}

for i, x_batch in enumerate(tqdm(data_loader_valid)):
    x_images = x_batch['image_id']
    x_batch = x_batch["image"]
    x_batch = x_batch.to(device, dtype=torch.float)

    with torch.no_grad():

        pred, embed = model(x_batch)
        pred = torch.sigmoid(pred)

        for x, y in zip(x_images, pred):
            test_pred_dict[x] = y.detach().cpu().numpy()
        for x, y in zip(x_images, embed):
            e = y.squeeze().detach().cpu().numpy()
            e = np.expand_dims(e, axis=0)
            test_embed_dict[x] = pca.transform(e)[0]

test_embed_df = pd.DataFrame.from_dict(test_embed_dict, orient='index')
test_embed_df.to_csv('test_embeds.csv')

test_pred_df = pd.DataFrame.from_dict(test_pred_dict, orient='index')
test_pred_df.to_csv('test_preds.csv')
