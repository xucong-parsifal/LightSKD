import torch
from backbone.resnet import *
from backbone.ResNeXt import resnext50_32x4d
from backbone.densenet import densenet121 as densenet121

def output_process(output):
    return torch.sort(output)[0]

def params_detection(net):
    stat = net.state_dict()
    for k, v in stat.items():
        try:
            print(k, v.mean(), v.std(), v.max(), v.min())
        except:
            print(k, v)

def get_model(args):
    if args.model == "ResNet18":
        return ResNet18(num_class=args.num_classes)
    elif args.model == "ResNet50":
        return ResNet50(num_class=args.num_classes)
    elif args.model == "ResNet101":
        return ResNet101(num_class=args.num_classes)
    elif args.model == "ResNext":
        return resnext50_32x4d(args.num_classes)
    elif args.model == "RenseNRet":
        return densenet121(num_classes=args.num_classes)


class var_metrics():
    def __init__(self,device):
        self.list = []
        self.device = device
        self.now_var = torch.zeros((120)).to(device)
        self.now_list = []
        self.cnt = 0

    def add_ranked(self,outputs):
        with torch.no_grad():
            self.cnt += len(outputs)
            for item in outputs:
                self.now_var += item.to(self.device)
                self.now_list.append(item)

    def average(self):
        with torch.no_grad():
            self.now_var = self.now_var / self.cnt
            temp = 0
            for item in self.now_list:
                a = torch.norm(item-self.now_var)
                temp += a*a
            temp = torch.sqrt(temp/self.cnt)
            print(temp)
            self.list.append(temp)
            self.cnt = 0
            self.now_var = torch.zeros((200)).to(self.device)
            self.now_list = []

    def write_in(self):
        f = open("./logs/variance-log.txt","w")
        for item in self.list:
            f.write(str(item)+"\n")
        f.close()

print("\033[1;31;40mhaha")