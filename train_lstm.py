import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from datasets import PredictionsDataset
from models import EmbeddingSmootherModel

COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

stage_dir = './stage_2'
metadata_csv = './rsna-intracranial-sequence-metadata'
working_directory = './pca-x120'

train = pd.read_csv(os.path.join(stage_dir, 'stage_2_train.csv'))
test = pd.read_csv(os.path.join(stage_dir, 'stage_2_sample_submission.csv'))

test_preds = pd.read_csv(f'{working_directory}/test_preds.csv', index_col=False)
valid_preds = pd.read_csv(f'{working_directory}/valid_preds.csv', index_col=False)
train_preds = pd.read_csv(f'{working_directory}/train_preds.csv', index_col=False)

test_preds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
valid_preds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
train_preds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)

valid_embeds = pd.read_csv(f'{working_directory}/valid_embeds.csv', index_col=False)
train_embeds = pd.read_csv(f'{working_directory}/train_embeds.csv', index_col=False)
test_embeds = pd.read_csv(f'{working_directory}/test_embeds.csv', index_col=False)

valid_embeds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
train_embeds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
test_embeds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)

FEATURES = train_embeds.columns.size - 1

test_metadata_noidx = pd.read_csv(f'{metadata_csv}/test_metadata_noidx.csv')
train_metadata_noidx = pd.read_csv(f'{metadata_csv}/train_metadata_noidx.csv')

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']

# Also prepare the test data
test[['ID', 'Image', 'Diagnosis']] = test['ID'].str.split('_', expand=True)
test['Image'] = 'ID_' + test['Image']
test = test[['Image', 'Label']]
test.drop_duplicates(inplace=True)

merged_train = pd.merge(left=train_preds, right=train_metadata_noidx, how='left', left_on='Image', right_on='ImageId')
merged_valid = pd.merge(left=valid_preds, right=train_metadata_noidx, how='left', left_on='Image', right_on='ImageId')
merged_test = pd.merge(left=test_preds, right=test_metadata_noidx, how='left', left_on='Image', right_on='ImageId')

merged_train = pd.merge(left=merged_train, right=train, how='left', left_on='Image', right_on='Image')
merged_valid = pd.merge(left=merged_valid, right=train, how='left', left_on='Image', right_on='Image')
merged_test = pd.merge(left=merged_test, right=test, how='left', left_on='Image', right_on='Image')

merged_train = pd.merge(left=merged_train, right=train_embeds, how='left', left_on='Image', right_on='Image')
merged_valid = pd.merge(left=merged_valid, right=valid_embeds, how='left', left_on='Image', right_on='Image')
merged_test = pd.merge(left=merged_test, right=test_embeds, how='left', left_on='Image', right_on='Image')

train_dataset = PredictionsDataset(merged_train, COLS, 120, True, None)
valid_dataset = PredictionsDataset(merged_valid, COLS, 120, True, None)
test_dataset = PredictionsDataset(merged_test, COLS, 120, False, None)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_model = EmbeddingSmootherModel(120, 256)

embedding_model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
plist = [
    {'params': embedding_model.parameters(), 'lr': 3e-5},
]
optimizer = optim.Adam(plist, lr=3e-5)

n_epochs = 5
for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    print('-' * 10)

    embedding_model.train()
    tr_loss = 0.
    st_loss = 0.

    auc_preds = []
    auc_truths = []

    # tk0 = tqdm(data_loader_train, desc="Iteration")

    for step, batch in enumerate(data_loader_train):
        inputs = batch["preds"]
        labels = batch["labels"]
        embeds = batch["concats"]
        if step == 0:
            print(embeds.shape)

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        embeds = embeds.to(device, dtype=torch.float)

        outputs = embedding_model(embeds, inputs)
        loss = criterion(outputs.view(-1, 6), labels.view(-1, 6))

        tr_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step % 1048 == 0:
            epoch_loss = tr_loss / (step + 1)
            st_epoch_loss = 0
            print('Training Loss at {}: {:.4f}\t{:.4f}'.format(step, epoch_loss, st_epoch_loss))

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))

    embedding_model.eval()
    tr_loss = 0.
    st_loss = 0.

    auc_preds = []
    auc_truths = []
    auc_preds_individual = []
    auc_truths_individual = []

    for step, batch in enumerate(data_loader_valid):
        inputs = batch["preds"]
        labels = batch["labels"]
        embeds = batch["concats"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        embeds = embeds.to(device, dtype=torch.float)

        outputs, scan_output = embedding_model(embeds, inputs)

        tr_loss += criterion(outputs.view(-1, 6), labels.view(-1, 6)).item()

        outputs = torch.sigmoid(outputs)

        detached_outputs = outputs.view(-1, 6).detach().cpu().numpy()
        detached_labels = labels.view(-1, 6).detach().cpu().numpy()

        auc_preds.append(np.max(detached_outputs, axis=0))
        auc_truths.append(np.max(detached_labels, axis=0))

        for o, l in zip(detached_outputs, detached_labels):
            auc_preds_individual.append(o)
            auc_truths_individual.append(l)

    epoch_loss = tr_loss / len(data_loader_valid)

    roc_preds = np.array(auc_preds)
    roc_truths = np.array(auc_truths)

    roc_preds_individual = np.array(auc_preds_individual)
    roc_truths_individual = np.array(auc_truths_individual)

    print('Validation Loss: {:.4f}'.format(epoch_loss))
    print('-----------SCAN-----------')
    print('-----------ROCAUC-----------')
    for tp in range(0, 6):
        print(COLS[tp], roc_auc_score(roc_truths[:, tp], roc_preds[:, tp]), )
    print('-----------F1/Sens/Spec-----------')
    for tp in range(0, 6):
        print(COLS[tp])
        print(classification_report(roc_truths[:, tp], np.round(roc_preds[:, tp]), digits=4))
    print('---------------------------------')
    print('-----------SLICE-----------')
    print('-----------ROCAUC-----------')
    for tp in range(0, 6):
        print(COLS[tp], roc_auc_score(roc_truths_individual[:, tp], roc_preds_individual[:, tp]), )
    print('-----------F1/Sens/Spec-----------')
    for tp in range(0, 6):
        print(COLS[tp])
        print(classification_report(roc_truths_individual[:, tp], np.round(roc_preds_individual[:, tp]), digits=4))
    print('---------------------------------')

# Submission
cols = COLS
results = []

tk0 = tqdm(data_loader_test, desc="Iteration")
embedding_model.eval()
for step, batch in enumerate(tk0):
    seriesId = test_dataset.series[step]
    images = test_dataset.data[test_dataset.data['SeriesInstanceUID'] == seriesId].sort_values(
        by=['ImagePositionSpan', 'ImageId']).ImageId.to_numpy()
    x_batch = batch["embeds"]
    x_batch_x = batch["preds"]
    with torch.no_grad():
        preds = embedding_model(x_batch.to(device, dtype=torch.float), x_batch_x.to(device, dtype=torch.float))
        preds = torch.sigmoid(preds)

        preds = preds.detach().cpu().numpy()[0]
        for img, pred in zip(images, preds):
            res = {
                'Image': img
            }
            for x, y in zip(cols, pred):
                res[x] = y
            results.append(res)

sub_df = pd.DataFrame(results)

submission = pd.read_csv(os.path.join('stage_2', 'stage_2_sample_submission.csv'))

melt_df = sub_df.melt(id_vars=['Image'], var_name='Diagnosis', value_name='Label')
melt_df['ID'] = melt_df['Image'] + '_' + melt_df['Diagnosis']
melt_df = melt_df.drop(['Image', 'Diagnosis'], axis=1)

sub_df = submission.merge(melt_df, on='ID', how='inner', suffixes=('_x', '')).drop('Label_x', axis=1)
sub_df.to_csv('embedding_sub.csv', index=False)

first_df = sub_df.copy()
