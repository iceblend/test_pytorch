import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

random.seed(777)
torch.manual_seed(777)

training_epochs = 30
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)


# MNIST data image of shape 28 * 28 = 784
linear = nn.Linear(784, 10, bias=True).to(device)

# 비용함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수 포함.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)


for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 저장함
    avg_cost = 0
    total_batch = len(data_loader)
    
    for X, Y in data_loader:
        # 배치 크기가 100 이므로 아래의 연산에서 X는
        # (100, 784)의 텐서가 된다.
        
        X = X.view(-1, 28 * 28).to(device)
        # 레이블은 원핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')


# 테스트 데이터를 사용하여 모델을 테스트 한다.
with torch.no_grad(): #gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accruacy : ', accuracy.item())
    
    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다.
    r = random.randint(0, len(mnist_test) - 1)
    X_Single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_Single_data = mnist_test.test_labels[r:r + 1].to(device)
    
    print('Label : ', Y_Single_data.item())
    single_prediction = linear(X_Single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())
    
    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()











