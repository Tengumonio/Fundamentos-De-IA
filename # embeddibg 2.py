# embeddibg 2


import torch    
import torch.nn as nn 
from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

inputs = torch.tensor([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])

labels = torch.tensor([[0., 1., 0., 0.],
                       [0., 0., 1., 0.,],
                       [0., 0., 0., 1.],
                       [0., 1., 0., 0.]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

#Embedding con linear
class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):
        super().__init__()
        
        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False) 
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        hidden = self.input_to_hidden(input)
        output_values = self.hidden_to_output(hidden)    
        return(output_values)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i)
        return loss
    
#crear la red

modelLinear = WordEmbeddingWithLinear()

#mostrar  parametros antes del aprendizaje

data = {
    "w1": modelLinear.input_to_hidden.weight.detach()[0].numpy(),
    "w2": modelLinear.input_to_hidden.weight.detach()[1].numpy(),
    "token": ["Dunas2", "es", "grandiosa", "Godzilla"],
    "input": ["input1", "input2", "input3", "input4"]
}

df = pd.DataFrame(data)
df

# Graficar con scatterplot 

sns.scatterplot(data=df, x="w1", y="w2")

plt.text(df.w1[0], df.w2[0], df.token[0],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[3], df.w2[3], df.token[3],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.show()

#entrrenamiento 

trainer = L.Trainer(max_epochs=5000)
trainer.fit(modelLinear, train_dataloaders=dataloader)

data = {
    "w1": modelLinear.input_to_hidden.weight.detach()[0].numpy(),
    "w2": modelLinear.input_to_hidden.weight.detach()[1].numpy(),
    "token": ["Dunas2", "es", "grandiosa", "Godzilla"],
    "input": ["input1", "input2", "input3", "input4"]
}

df = pd.DataFrame(data)
df

# Graficar con scatterplot 

sns.scatterplot(data=df, x="w1", y="w2")

plt.text(df.w1[0]-0.2, df.w2[0]+0.1, df.token[0],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[3]-0.3, df.w2[3], df.token[3],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.show()

#++++++++++++++
# Se pueden poner los resultados en un objeto
# Embedding de pytorh pero se le tienen que dar
# tranpuestos
#print(modelLineal.input_to_hidden.weight)

word_embeddings = nn.Embedding.from_pretrained(modelLinear.input_to_hidden.weight.T)
vocab = {'Dunas2':0,
         'es':1,
         'grandiosa':2,
         'Godzilla':3}
print(word_embeddings(torch.tensor(vocab['Dunas2'])))



    
