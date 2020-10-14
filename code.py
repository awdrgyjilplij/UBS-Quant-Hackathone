import alphien
import torch
import torch.nn as nn

dl = alphien.data.DataLoader()
dataGen = dl.batch()
data = next(dataGen)

input_size = 164
output_size = 1
learning_rate = 0.001


def engineerData(df, engineeringParam1='defaultValue', engineeringParam2='anotherValue', isTrain=True):
    # ...
    # do some processing of df in here.
    # as this is a toy e.g., I just return data without any changes
    newEngineeredData = df
    # ...
    return newEngineeredData


model = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

trainTestSplit = 0.75
dataGen = dl.batch(fromRow=1, toRow=5000000)

print("Let's train!")
model.train()
for epoch, data in enumerate(dataGen):
    trainTestIndex = int(trainTestSplit * data.shape[0])
    train = engineerData(data.iloc[:trainTestIndex, :])

    inputs = torch.tensor(train.iloc[:, :-1].values.astype(float)).float()
    targets = torch.tensor(
        train.iloc[:, -1].values.astype(float)).unsqueeze(1).float()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print('Epoch [{}], Loss: {:.4f}'.format(epoch + 1, loss.item()))

print("Let's predict!")
model.eval()
unseen = dl.batch(fromRow=5000000)
for data in unseen:
    trainTestIndex = int(trainTestSplit * data.shape[0])
    train = engineerData(data.iloc[trainTestIndex:, :])

    inputs = torch.tensor(train.iloc[:, :-1].values.astype(float)).float()
    targets = torch.tensor(
        train.iloc[:, -1].values.astype(float)).unsqueeze(1).float()
    pred = model(inputs)
    print("Loss of prediction: ", criterion(pred, targets).detach().item())

print("Done")

def dataTransformFunc(data):
    return torch.tensor(engineerData(
        data.iloc[:, :].astype(float), isTrain=False).values).float()

def myPredictFunc(newData, dataTransformFunc, model):
    inputs = dataTransformFunc(newData) 
    return model(inputs)