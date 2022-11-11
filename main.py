import torch
import torch.nn as nn

from src.datasets.dataset import Dataset
from src.model.LSTM import LSTM
from src.trainer.Trainer import Trainer
from src.trainer.utils import predict

filename = '.inputs/spa-eng/spa.txt'
data = Dataset(filename, 'english')
data.split_dataset(20000, 2000, 2000)
data.define_dataloaders_and_sizes()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.001
dropout = .5
num_epochs = 15
hidden_size = 128

lstm = LSTM(data.n_unique_words+2, hidden_size,
            data.n_unique_words+2, dropout, device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(),
                             lr=learning_rate,
                             weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

trainer = Trainer('test3', lstm, data.dataloaders, data.dataset_sizes,
                  criterion, optimizer, scheduler, device, num_epochs)

trainer.run()
trainer.plot_training_curves()
trainer.save()

print(predict(lstm, data.language_dictionary, data.language_list, "what is", device))
print(predict(lstm, data.language_dictionary, data.language_list, "my name", device))
print(predict(lstm, data.language_dictionary, data.language_list, "how are", device))
print(predict(lstm, data.language_dictionary, data.language_list, "hi", device))
print(predict(lstm, data.language_dictionary, data.language_list, "choose", device))

