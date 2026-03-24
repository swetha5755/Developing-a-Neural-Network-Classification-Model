# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="630" height="687" alt="image" src="https://github.com/user-attachments/assets/2456efeb-786f-4bdc-88fd-7e18ad71fcc0" />


## DESIGN STEPS
### STEP 1: Load and Preprocess Data

Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).



### STEP 2: Feature Scaling and Data Split

Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance



### STEP 3: Convert Data to PyTorch Tensors

Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.



### STEP 4: Define the Neural Network Model

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.



### STEP 5: Train the Model

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.



### STEP 6: Evaluate and Predict

Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.






## PROGRAM

### Name: Swetha S

### Register Number:212224040344

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)
        
    def forward(self, x):
       x=F.relu(self.fc1(x))
       x=F.relu(self.fc2(x))
       x=F.relu(self.fc3(x))
       x=self.fc4(x)
       return x

# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
      model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()




    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


```

### Dataset Information
<img width="1087" height="205" alt="image" src="https://github.com/user-attachments/assets/7c6c6b57-b563-4d2f-b8fa-cc0ae9c852ae" />


### OUTPUT

## Confusion Matrix

<img width="756" height="597" alt="image" src="https://github.com/user-attachments/assets/f5cdf619-0d18-43f1-86e8-7af3469d5f26" />


## Classification Report
<img width="462" height="341" alt="image" src="https://github.com/user-attachments/assets/775590ec-ab20-47ef-9dfb-a12266b52006" />


### New Sample Data Prediction
<img width="294" height="76" alt="image" src="https://github.com/user-attachments/assets/e8736beb-b207-4f2b-8ffb-f0681d2ede01" />


## RESULT
This program has been executed successfully.
