import time
import copy
import torch
import numpy as np

import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, name, model, dataloaders, dataset_sizes,
                 criterion, optimizer, scheduler, device,
                 num_epochs=25):
        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.model_name = name

        self.phases = ['train', 'val', 'test']

        self.best_model_wts = copy.deepcopy(model.state_dict())  # keep the best weights stored separately
        self.best_loss = np.inf
        self.best_epoch = 0

        self.training_curves = {}

        self._init_training_curves()

    def _init_training_curves(self):
        for phase in self.phases:
            self.training_curves[phase + '_loss'] = []

    def _set_model_mode(self, phase: str):
        self.model.train() if phase == 'train' else self.model.eval()

    def run(self):
        since = time.time()
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            for phase in self.phases:
                self._set_model_mode(phase)

                running_loss = 0.0

                # Iterate over data
                for input_sequence, target_sequence in self.dataloaders[phase]:
                    # Now Iterate through each sequence here:

                    hidden = self.model.init_hidden()  # Start with a fresh hidden state

                    current_input_sequence = input_sequence.to(self.device)
                    current_target_sequence = target_sequence.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        loss = 0
                        # Make a prediction for each element in the sequence,
                        # keeping track of the hidden state along the way
                        for i in range(current_input_sequence.size(0)):
                            # Need to be clever with how we transfer our hidden layers to the device
                            current_hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
                            output, hidden = self.model(current_input_sequence[i], current_hidden)
                            loss_aux = self.criterion(output, current_target_sequence[i])
                            loss += loss_aux

                        # backward + update weights only if in training phase at the end of a sequence
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() / current_input_sequence.size(0)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                self.training_curves[phase + '_loss'].append(epoch_loss)

                print(f'{phase:5} Loss: {epoch_loss:.4f}')

                # deep copy the model if it's the best loss
                # Note: We are using the train loss here to determine our best model
                if phase == 'train' and epoch_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = epoch_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {self.best_loss:4f} at epoch {self.best_epoch}')

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)

    def plot_training_curves(self, metrics=['loss']):
        epochs = list(range(len(self.training_curves['train_loss'])))
        for metric in metrics:
            plt.figure()
            plt.title(f'Training curves - {metric}')
            for phase in self.phases:
                key = phase + '_' + metric
                if key in self.training_curves:
                    plt.plot(epochs, self.training_curves[key])
            plt.xlabel('epoch')
            plt.legend(labels=self.phases)
            plt.savefig(f'.outputs/images/lstm_{self.model_name}-{metric}.png')
            plt.show()

    def save(self):
        torch.save(self.model, f'.outputs/models/{self.model_name}/model_weights.pt')