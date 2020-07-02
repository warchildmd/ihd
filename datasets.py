# from apex import amp
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img


def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return bsb_img


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            dicom = pydicom.dcmread(self.path + self.data.loc[idx, 'Image'] + '.dcm')
            img = bsb_window(dicom)
        except:
            img = np.zeros((512, 512, 3))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        if self.labels:

            labels = torch.tensor(
                self.data.loc[
                    idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}

        else:

            return {'image': img}


class PredictionsDataset(Dataset):

    def __init__(self, data, col_names, features=120, train=True, series=None):
        self.data = data
        self.train = train
        self.col_names = col_names
        self.embed_cols = [str(i) for i in range(features)]

        if series is None:
            self.series = self.data['SeriesInstanceUID'].unique()
        else:
            self.series = series

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        series_id = self.series[idx]
        images = self.data[self.data['SeriesInstanceUID'] == series_id].sort_values(by=['ImagePositionSpan', 'ImageId'])

        cols = self.col_names
        if self.train:
            cols = [x + '_x' for x in self.col_names]

        image_preds = images[cols].to_numpy().astype(np.float)

        if self.train:
            image_truths = images[[x + '_y' for x in self.col_names]].to_numpy().astype(np.float)

            image_embeds = images[self.embed_cols].to_numpy().astype(np.float)

            return {
                'preds': torch.tensor(image_preds).to(torch.float),
                'labels': torch.tensor(image_truths).to(torch.float),
                'embeds': torch.tensor(image_embeds).to(torch.float)
            }
        else:
            image_embeds = images[self.embed_cols].to_numpy().astype(np.float)
            return {
                'preds': torch.tensor(image_preds).to(torch.float),
                'embeds': torch.tensor(image_embeds).to(torch.float)
            }
