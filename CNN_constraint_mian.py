from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from CNN_constraint import CNN_constraint
from data_loader_channels import MyDataset
from utils import AverageMeter, accuracy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import cohen_kappa_score
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


chanel_num = 44
window_size = 7

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default= 1, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  

def main():
    global best_acc
    start_epoch = args.start_epoch

    dataset_train = MyDataset(datatxt='./demo_data/train_dataset_extend/data_list.txt', transform=transforms.ToTensor()) 
    dataset_test = MyDataset(datatxt='./demo_data/valid_dataset_extend/data_list.txt', transform=transforms.ToTensor()) 
    trainloader = data.DataLoader(dataset=dataset_train,batch_size=args.train_batch,shuffle=True,num_workers=0)
    testloader = data.DataLoader(dataset=dataset_test,batch_size=args.test_batch,shuffle=False,num_workers=0)
    num_classes = 2
   
    model = CNN_constraint(num_classes=num_classes)
    model = model.cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        print('Train Loss: %.4f |Test Loss:%.4f |Train Acc: %.4f|Test Acc: %.4f' % (train_loss,test_loss,train_acc.item(),test_acc.item()))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc.item())
        test_acc_list.append(test_acc.item())

    torch.save(model,'model_cnn_soft_hard_constraint.pth') ########

    
    x1t = range(0,args.epochs) 
    x2t = range(0,args.epochs) 
    y1t = train_loss_list
    y2t = test_loss_list
    y3t = train_acc_list
    y4t = test_acc_list
    plt.plot(x1t, y1t, '-',color='blue',label='Train loss')
    plt.plot(x1t, y2t, '-',color='red',label='Test loss')
    plt.xlabel('Iteration',family='Times New Roman',fontsize=12) 
    plt.ylabel('Loss',family='Times New Roman',fontsize=12)
    plt.xticks(fontname="Times New Roman", fontsize=10)
    plt.yticks(fontname="Times New Roman", fontsize=10)
    plt.legend(['Train loss','Test loss'],prop={"family": "Times New Roman", "size": 12}) 
    plt.show()

    plt.plot(x2t, y3t, '-',color='blue',label='Train accuracy')
    plt.plot(x2t, y4t, '-',color='red',label='Test accuracy')
    plt.xlabel('Iteration',family='Times New Roman',fontsize=12) 
    plt.ylabel('Accuracy',family='Times New Roman',fontsize=12) 
    plt.xticks(fontname="Times New Roman", fontsize=10)
    plt.yticks(fontname="Times New Roman", fontsize=10)
    plt.legend(['Train accuracy','Test accuracy'], prop={"family": "Times New Roman", "size": 12})
    plt.show()
    dataframe = pd.DataFrame({'train_loss':train_loss_list,'test_loss':test_loss_list,
                'train_acc':train_acc_list,'test_acc':test_acc_list})

    dataframe.to_csv("loss_acc.csv",index=False,sep=',')


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
  
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()


    for batch_idx, (inputs, term, term1, term2, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, term, term1, term2, targets = inputs.cuda(), term.cuda(), term1.cuda(), term2.cuda(), targets.cuda(non_blocking=True)

        inputs, term, term1, term2, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(term), torch.autograd.Variable(term1), torch.autograd.Variable(term2), torch.autograd.Variable(targets)
  
        predict_result, x_term1 = model(inputs,term, term1, term2)
        main_loss = criterion(predict_result, targets)
        loss = main_loss
        g_loss = ((nn.Softmax()(predict_result)[:,1]-x_term1[:,0,0,0])**2).sum()/2 
        loss = main_loss + g_loss
        
        prec1 = accuracy(predict_result.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    for batch_idx, (inputs, term, term1, term2, targets) in enumerate(testloader):
        if use_cuda:
            inputs, term, term1, term2, targets = inputs.cuda(), term.cuda(), term1.cuda(), term2.cuda(), targets.cuda()
        inputs, term, term1, term2, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(term), torch.autograd.Variable(term1), torch.autograd.Variable(term2), torch.autograd.Variable(targets)

        with torch.no_grad():

            predict_result,x_term1 = model(inputs,term, term1, term2)
            main_loss = criterion(predict_result, targets)
            loss = main_loss
            g_loss = ((nn.Softmax()(predict_result)[:,1]-x_term1[:,0,0,0])**2).sum()/2
            loss = main_loss + g_loss
            
        prec1 = accuracy(predict_result.data, targets.data, topk=(1, ))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))


    return (losses.avg, top1.avg)

def test_with_model(model_name):
    model = torch.load(model_name)
    model.eval()
    test_loader = MyDataset(datatxt='./demo_data/valid_dataset_extend/data_list.txt', transform=transforms.ToTensor())

    correct = 0
    total = 0
    TN=0
    TP=0
    FP=0
    FN=0
    predict_value = []
    target_value = []
    probability_value=[]

    with torch.no_grad():
        for batch_num, (data, term, term1, term2, target) in enumerate(test_loader):
            if use_cuda:
                target = torch.tensor(target)
                data, term,term1, term2, target = data.cuda(), term.cuda(), term1.cuda(), term2.cuda(), target.cuda(non_blocking=True)
            data = data.view(1,chanel_num,window_size,window_size)
            term = term.view(1,1,window_size,window_size)
            term1 = term1.view(1,1,window_size,window_size)
            term2 = term2.view(1,1,window_size,window_size)
            target=torch.tensor([target])
            output, x_term1 = model(data,term, term1, term2)

            probability = nn.functional.softmax(output,dim=1)
            prediction = torch.max(output, 1)

            total += target.size(0)
            if (target.cpu().numpy()==0):
                if (prediction[1].cpu().numpy()==0):
                        TN = TN+1

            if (target.cpu().numpy()==0):
                    if (prediction[1].cpu().numpy()==1):
                        FP = FP+1

            if (target.cpu().numpy()==1):
                    if (prediction[1].cpu().numpy()==0):
                        FN = FN+1

            if (target.cpu().numpy()==1):
                if (prediction[1].cpu().numpy()==1):
                        TP = TP+1

            correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            pred=prediction[1].cpu().numpy()
            predict_value.append(pred[0])

            targ=target.cpu().numpy()
            target_value.append(targ[0])


            probab=probability.cpu().numpy()
            probability_value.append(probab[:,1])


    print('ACC :%.3f %d/%d' % (correct / total * 100, correct, total))
    print('TN:',TN)
    print('FP:',FP)
    print('FN:',FN)
    print('TP:',TP)

    kappa = cohen_kappa_score(np.array(target_value).reshape(-1,1), np.array(predict_value).reshape(-1,1))
    print('Kappa：',kappa)

    fpr, tpr, thresholds = roc_curve(target_value, probability_value, drop_intermediate=False)

    AUC = auc(fpr, tpr)
    print("AUC : ", AUC)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='CNN({:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
    test_with_model('model_cnn_soft_hard_constraint.pth')