import os, pickle

from physics.simulation import mcfm, msq
from physics.analysis import zz4l
from physics.hstar import c6

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch

class AliceDataModule(L.LightningDataModule):

    def __init__(self, filepath: str = '', features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG, scaler_path = 'scaler.pkl', c6_points = None, sample_size = 10000, batch_size: int = 32, random_state: int=None) -> None:
        super().__init__()

        self.filepath = filepath
        self.features = features
        self.numerator_component = numerator_component
        self.denominator_component = denominator_component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scaler_path = scaler_path
        self.c6_points = c6_points
    
    def prepare_data(self):
        events = mcfm.from_csv(cross_section=1.0, file_path=self.filepath)
        self.events = zz4l.analyze(events)

    def setup(self, stage: str):
        if stage=='fit':
            events_train, events_val = self.events.shuffle(random_state=self.random_state).split(train_size=0.5, val_size=0.5)

            if self.c6_points is None:
                self.training_data = JointLikelihoodDataset(events_train, features=self.features, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)
                self.validation_data = JointLikelihoodDataset(events_val, features=self.features, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)
            else:
                self.training_data = JointLikelihoodParameterizedDataset(events_train, features=self.features, c6_points=self.c6_points, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)
                self.validation_data = JointLikelihoodParameterizedDataset(events_val, features=self.features, c6_points=self.c6_points, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)

            # Apply Scaler to both datasets after fitting to training data
            self.training_data.X = self.scaler.fit_transform(self.training_data.X)
            if self.scaler_path is not None:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            self.validation_data.X = self.scaler.transform(self.validation_data.X)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

class JointLikelihoodDataset(Dataset):

    def __init__(self, events, features, sample_size, numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG, random_state=None):
        super().__init__()
        events = events.unweight(sample_size, random_state=random_state)

        # Get only required features
        self.X = events.kinematics[features].to_numpy()

        # Get PDF ratios for p(theta_0)/p(theta_1)
        r = events.probabilities/events.reweight(numerator=numerator_component, denominator=denominator_component).probabilities

        self.s = (1/(1 + r)).to_numpy()

    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)
    
class JointLikelihoodParameterizedDataset(Dataset):

    def __init__(self, events, features, c6_points, sample_size, numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG, random_state=None):
        super().__init__()
        c6_mod = c6.Modifier(baseline=numerator_component, events=events, c6_points=[-5,-1,0,1,5]) if numerator_component != msq.Component.INT else c6.Modifier(baseline=numerator_component, events=events, c6_points=[-5,0,5])
        _, c6_probabilities = c6_mod.modify(c6_points)

        X = np.tile(events.kinematics[features].to_numpy(), (len(c6_points),1))
        c6_column = np.repeat(c6_points, len(events.kinematics), axis=0)[:,np.newaxis]

        X = np.concatenate([X, c6_column], axis=1)

        probabilities_numerator = c6_probabilities.flatten()
        probabilities_denominator = np.tile(events.probabilities.to_numpy(), len(c6_points))

        sample_weights = pd.Series((probabilities_numerator + probabilities_denominator)/2*sample_size).reset_index(drop=True)
        
        unweighted_indices = sample_weights.sample(n=sample_size, replace=True, weights=sample_weights, random_state=random_state).index

        s = 1/(1+probabilities_denominator/probabilities_numerator)

        self.X = X[unweighted_indices]
        self.s = s[unweighted_indices]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)
