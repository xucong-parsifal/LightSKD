import os
from torch import nn, optim
from datasets import get_trainloader as tl, get_testloader as tel
from backbone.resnet import ResNet18,ResNet50,ResNet101
from backbone.adapter import adapter_2
from backbone.mobilenetv2 import *
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(description="KPID teacher training")
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--datasets", default='CIFAR100', type=str)
parser.add_argument("--temperature", default=4.0, type=float)
parser.add_argument("--alpha", default=20, dest="intro-class loss")
parser.add_argument("--beta", default=20, dest="time wise loss")
parser.add_argument("--data_path", default='./datasets/')
parser.add_argument("--model", default='resnet18')
parser.add_argument("--model_path", default="./save/")
parser.add_argument("--initial_lr", default=0.1, type=float)
args = parser.parse_args()

trainloader = tl(params=args.datasets)
testloader = tel(params=args.datasets)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def output_process(output):
    return torch.sort(output)[0]

def params_detection(net):
    stat = net.state_dict()
    for k, v in stat.items():
        try:
            print(k, v.mean(), v.std(), v.max(), v.min())
        except:
            print(k, v)

def get_model():
    if args.model == "resnet18":
        return ResNet18()
    elif args.model == "resnet50":
        return ResNet50()
    elif args.model == "resnet101":
        return ResNet101()

net = get_model()
net_d = adapter_2()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params': net.parameters()},{'params': net_d.parameters()}], lr=0.1, momentum=0.9, weight_decay=5e-4)

load_dir = './save/'
load_name = load_dir+str(args.model)+'.pth'
if not os.path.isfile(load_dir):
    os.makedirs(load_dir)
if os.path.isfile(load_name):
    checkpoint = torch.load(load_name,map_location=device)
    net.load_state_dict(checkpoint['net'])
    net_d.load_state_dict(checkpoint['net_d'])
net = net.to(device)
net_d = net_d.to(device)
    
def train(epoch):
    global init, similarity, similarity_chosed, trainloader, testloader
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_i_loss = 0
    train_r_loss = 0
    last_output_processed= torch.rand((128, 100))
    last_output = None
    last_data = None
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # learn labels
        outputs,argen = net(inputs[:,0,...])
        output1,argen1 = net(last_data)
        outputs_d = net_d(argen[1])
        
        # HL-LOSS
        loss = torch.mean(criterion(outputs, targets))+torch.mean(criterion(outputs_d, targets))
        loss += 100
        T =4

        # RI-LOSS
        ist = 0.2*nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs/T,dim=1), F.softmax((outputs_d.detach())/T,dim=1))* (T * T)
        loss += ist
        train_i_loss += ist.item()
        output_processed = output_process(outputs)

        #SW-LOSS
        if batch_idx  !=0:
            T =4
            # adv loss
            adv_loss = nn.KLDivLoss(reduction="batchmean") (F.log_softmax(output_processed/T,
               dim=1),F.softmax((last_output_processed.detach())/T,
               dim=1))* (T * T)+nn.KLDivLoss(reduction="batchmean")(F.log_softmax(output1,
               dim=1),F.softmax((last_output.detach())/T,dim=1))* (T * T)
            loss += adv_loss
            train_r_loss += adv_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().float().cpu()
        train_loss += loss.item()
        
        last_output_processed = output_processed
        last_output = outputs
        last_data = inputs[:,1,...]
        
        if (batch_idx + 1) % 40 == 0:
            print(batch_idx + 1, len(trainloader),
                  'Loss: %.3f ---------- Accuracy: %.3f%% (%d/%d) ------- istr: %.6f ------- sort: %.6f'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                     train_i_loss / (batch_idx + 1),train_r_loss / (batch_idx + 1)))
    
    filename = './save/'+str(args.model)+'.pth'
    state = {'net':net.state_dict(),'net_d':net_d.state_dict()}

    torch.save(state, filename)



def etest():
    print("testing")
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,argen = net(inputs)
            correct += outputs.max(1)[1].eq(targets).sum()
            test_loss += torch.mean(criterion(outputs, targets))
            total += outputs.size(0)
        print(len(testloader),
              'Total Loss: %.3f | Average Accuracy: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


for epoch in range(0,200):
    if epoch in [60,120,160]:
        for params in optimizer.param_groups:
            params["lr"] /= 5
    train(epoch)
    etest()






