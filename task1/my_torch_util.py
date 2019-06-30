#encoding: utf-8
#file: my_torch_util.py
#author: shawn233
#date: 19-06-26

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report


def iterate_minibatch(x, y, batch_size):
    num_batches = int(np.ceil(len(x) / batch_size))
    for i in range(num_batches):
        end_pos = min((i+1)*batch_size, len(x))
        yield i, x[i*batch_size:end_pos], y[i*batch_size:end_pos]

def get_probs(net, x, device=torch.device("cpu:0")):
    '''
    Get probabilities given model and input
    In my model definition (models.py), the last layer
    does not have softmax activation. Therefore, the 
    probabilities should be estimated by taking softmax 
    of the output of net 
    '''
    x = x.to(device)
    return F.softmax(net(x), dim=1)

def get_preds(net, x):
    '''
    Get final predictions given model and input
    This is implemented by argmax after get_probs function
    '''
    return torch.argmax(get_probs(net, x), dim=1)

def train_model(\
    net, train_x, train_y,\
    test_x=None, test_y=None,\
    learning_rate=1e-3,\
    batch_size=100,\
    n_epochs=50,\
    weight_decay=1e-8,\
    device=torch.device("cpu:0")\
):
    net.to(device)
    train_x = torch.tensor(train_x).float().to(device)
    train_y = torch.tensor(train_y).float().to(device)

    if test_x is not None and test_y is not None:
        # [learn from Shokri] remove extra test samples
        if len(test_x) > len(train_x):
            test_x = test_x[:len(train_x)]
            test_y = test_y[:len(train_x)]  
        test_x = torch.tensor(test_x).float().to(device)
        test_y = torch.tensor(test_y).float().to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        net.train()
        total_loss = 0.0
        correct = 0
        n_batches = int(np.ceil(len(train_x) / batch_size))
        for i, batch_x, batch_y in iterate_minibatch(train_x, train_y, batch_size):
            optimizer.zero_grad()
            output = net(batch_x)
            loss = criterion(output*batch_y.abs(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            #pred = F.softmax(output, dim=1).argmax(dim=1, keepdim=True)
            #correct += pred.eq(batch_y.view_as(pred)).sum().item()
        #train_acc = correct/len(train_y)
        print ("epoch [ {} / {} ] loss: {:.4f}".\
            format(epoch+1, n_epochs, total_loss/n_batches), end=' ')

        if test_x is not None and test_y is not None:
            net.eval()

            test_loss = 0.0
            correct = 0
            n_batches = int(np.ceil(len(test_y)/batch_size))
            with torch.no_grad():
                for i, batch_x, batch_y in iterate_minibatch(test_x, test_y, batch_size):
                    output = net(batch_x)
                    test_loss += criterion(output, batch_y).item()
                    #pred = F.softmax(output, dim=1).argmax(dim=1, keepdim=True)
                    #correct += pred.eq(batch_y.view_as(pred)).sum().item()
            #test_loss /= n_batches
            #test_acc = correct/len(test_y)
            print ("test_loss: {:.4f}".format(test_loss/n_batches), end='\n')
        else:
            print() # newline

    #test_and_report(net, train_x, train_y, batch_size=batch_size, device=device)
    #test_and_report(net, test_x, test_y, batch_size=batch_size, device=device)

def test_and_report(net, x, y, batch_size=128, device=torch.device("cpu:0")):
    pred = []
    for i, batch_x, batch_y in iterate_minibatch(x, y, batch_size):
        probs = get_probs(net, batch_x, device=device)
        pred.append(probs.argmax(dim=1))
    pred = torch.cat(pred).cpu().detach().numpy()
    print(classification_report(y.cpu(), pred))
    print()