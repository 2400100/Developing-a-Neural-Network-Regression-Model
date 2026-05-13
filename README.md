# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model

<img width="1095" height="745" alt="542725068-a10c57a4-bf02-4043-9a23-10da995e3d1c" src="https://github.com/user-attachments/assets/0afe7f51-a91e-4dc0-a799-bfc3ff7ad5b1" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SUBIKSHA K

### Register Number: 212224040332

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/deep learning  - Sheet1.csv')
X = dataset1[['input ']].values
y = dataset1[['output']].values
dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: SUBIKSHA K
# Register Number: 212224040332
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
lig = NeuralNet ()
criterion = nn.MSELoss ()
optimizer = optim.RMSprop(lig.parameters(),lr=0.001)

# Name: SUBIKSHA K
# Register Number:212224040332
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = ai_brain(X_train) # Get predictions from the model
    loss=criterion(outputs,y_train)
    loss.backward()
    optimizer.step()
    lig.history['loss'].append(loss.item())
    if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

            ai_brain.history['loss'].append(loss.item())
            if epoch % 200 == 0:
              print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(lig.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
<img width="366" height="643" alt="image" src="https://github.com/user-attachments/assets/7724b007-f0e1-4c7b-be97-9fa5ef572a94" />


### OUTPUT


<img width="216" height="33" alt="545352325-b7518663-d4f2-4d0f-b988-8ccc8aeef845" src="https://github.com/user-attachments/assets/f67f119a-d00c-41af-a0c8-12772af66ad7" />
<img width="375" height="221" alt="545352265-39e514ea-e199-45c8-96cf-e09c219a086d" src="https://github.com/user-attachments/assets/5db0c4d8-f4fd-4e13-ad2c-2fd622c6cc9b" />


### Training Loss Vs Iteration Plot

<img width="580" height="455" alt="download" src="https://github.com/user-attachments/assets/ce979c93-37a4-4ef2-957c-c3422c5a203d" />




### New Sample Data Prediction

<img width="297" height="31" alt="image" src="https://github.com/user-attachments/assets/15936d1f-5b5f-40ca-bb1b-1341317d6100" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
