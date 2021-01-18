from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.getModelnet_feat import GetDataset10, GetDataset40

from model.network import *
from model.tricenter_loss import TripletCenterLoss
# from model.center_loss import CenterLoss


from scipy.io import savemat
import itertools

'''

'''

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=256, help='batch size in training')
    parser.add_argument('--epoch',  default=80, type=int, help='number of epoch in training')
    parser.add_argument('--j',  default=4, type=int, help='number of epoch in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training SGD or Adam')
    parser.add_argument('--pretrained', dest='pretrained', action ='store_true', help='use pre-trained model')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--stage', type=str, default='train', help='train test exft(extract feature)')
    parser.add_argument('--model_name', type=str, default='attn_modelnet10', help='train test')
    parser.add_argument('--views',  default=12, type=int, help='the number of views')
    parser.add_argument('--nLayer',  default=6, type=int, help='the number of views')
    parser.add_argument('--num_classes',  default=40, type=int, help='the number of clsses')
    return parser.parse_args()


args = parse_args()
args.device = torch.device('cuda:%s'%args.gpu)



# 可以针对全局，也可以针对局部
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# use: 输入optimizer 和 epoch 就可以使用
#
def adjust_learning_rate(optimizers, epoch):
    """Sets the learning ra to the initial LR decayed by 10 every 200 epochs"""
    up = 1e-4; down=5e-6 
    lrs = (up- (down))*(np.cos([np.pi*i/args.epoch for i in range(args.epoch)])/2 + 0.5)+(down)

    
    for op in optimizers:
        op.param_groups[0]['lr'] = lrs[epoch]
    
    print('Learning Rate:%.6f' % lrs[epoch])
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    #     print ('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))
    return lrs[epoch]


def dist_elur(fts_q, fts_c):
    fts_q = fts_q/torch.norm(fts_q, dim=-1,keepdim=True)
    fts_c = fts_c/torch.norm(fts_c, dim=-1,keepdim=True)
    fts_qs = torch.sum((fts_q)**2,dim=-1,keepdim=True)
    fts_cs = torch.sum((fts_c)**2,dim=-1,keepdim=True).t()
    qc = torch.mm(fts_q,fts_c.t())
    dist = fts_qs + fts_cs - 2 * qc +1e-4
    return torch.sqrt(dist)


def dist_cos(fts_q, fts_c):
    up = torch.matmul(fts_q,fts_c.T)
    down1 = torch.sqrt(torch.sum((fts_q)**2,axis=-1,keepdims=True))
    down2  = torch.sqrt(torch.sum((fts_c)**2,axis=-1,keepdims=True).t())
    down = torch.mm(down1, down2)
    dist = up/(down+1e-4)
    return 1 - dist


def mi_dist(fts1,fts2,la1,la2,margin=1,mode=2):
    dist = dist_elur(fts1,fts2)
    index = (la1[:,mode].reshape(-1,1)==la2[:,mode].reshape(1,-1)).bool()
    ap = dist[index]
    lens = len(ap)
    an = torch.sort(dist[(1-index.long()).bool()])[0][:lens]
    if lens*2 > (dist.shape[0]**2):
        ap = ap.mean().unsqueeze(0)
        an = an.mean().unsqueeze(0)
    loss = nn.MarginRankingLoss(margin)(ap,an,torch.Tensor([-1]).to(device)    )
    return loss,ap.mean()


path_model = [
    'experiment/checkpoints/top.pth' # 0   训练最好的
]

path_mat =[os.path.join('metric', os.path.basename(i).split('pth')[0] + 'mat') for i in path_model]

def print_model_parm_nums(model): #得到模型参数总量 
    total = sum([param.nelement() for param in model.parameters()]) 
    print(' + Number of params: %.2fM' % (total )) #每一百万为一个单位 
    # print(' + Number of params: %.2fM' % (total / 1e6)) #每一百万为一个单位 
    return total

def main():
    global args
    # 数据记录
    logger_train = get_logger('%s_train'%(args.model_name))
    logger_test = get_logger('%s_test'%(args.model_name))
    top_acc = 0.0
    top_acc_path = ''
    acc_avg = AverageMeter()
    losses = ['loss_cls', 'loss_metric']



    # domainMode 0,1  是image, render/ mask  + train/test
    #           2,3   是image, render, mask  + 全部数据
    #           4,5   是image /render,      + train/test
    trainDataset =  GetDataset40(dataType='train')
    validateDataset =  GetDataset40(dataType='test')
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size =args.batchsize,shuffle=True,num_workers=args.j, pin_memory = True, drop_last=True)
    validateLoader = torch.utils.data.DataLoader(validateDataset, batch_size =args.batchsize, shuffle=False, num_workers=args.j,pin_memory=True,  drop_last=False)
    
    

    model = self_attn(args=args)
    model.eval()
    print_model_parm_nums(model)
    sys.exit(-1)
    
    net_metric = TripletCenterLoss(margin=5, num_classes=args.num_classes, feat_dim=1024)
    # print_model_parm_nums(net_metric)
    # net_metric = CenterLoss(num_classes = 40, feat_dim = 1024, device = args.device)

    if args.pretrained:
        if os.path.isfile(path_model[0]):
            load_model(model, path_model[0])
            print('! Using pretrained model')
        else: print('? No pretrained model!')
    else: print('! Do not use pretrained model')
    
    if args.gpu == '0,1':
        device_ids = [int(x) for x in args.gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    elif args.gpu == '0' or args.gpu == '1':
        model.to(args.device)
        net_metric.to(args.device)
        
        
    optimizers = []
    if args.optimizer == 'SGD':
            optimizers.append(torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9))
            optimizers.append(torch.optim.SGD(net_metric.parameters(), lr=args.lr, momentum=0.9))
            #optimizer_triCenter_c = torch.optim.SGD(itertools.chain(cri_triCenter_cat.parameters(),cri_triCenter_cent.parameters()), lr=0.1)
    elif args.optimizer == 'Adam':

            optimizers.append(torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.wd
            ))
            optimizers.append(torch.optim.Adam(
                net_metric.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.wd
            ))
    
    path = 'experiment/checkpoints/xx.pth'
    if args.stage=='exft':  # extract feature
        print('Extract feature ...')
        extract_feat(args,  model.eval(), path,  'metric/model_feat.npy', validateLoader)
        exit(-1)
    


    # train
    for epoch in range(0, args.epoch):
        cur_lr = adjust_learning_rate(optimizers,epoch)
        ftsa = []
        laa = []
        acc_avg.reset()
        times = [] 
        for idx, input_data in enumerate(tqdm(trainLoader)):
            data = input_data['data'].to(args.device)
            target = input_data['target'].reshape(-1).to(args.device)
            model.train()
            tp = time.time()
            out, fts = model(data)
            loss_cls = F.cross_entropy(out, target)
            times.append(time.time()-tp)
            loss_metric = net_metric(fts, target)
            if isinstance(loss_metric, tuple): loss_metric = loss_metric[0]
            loss = loss_cls  + 0.01* loss_metric
            
            for op in optimizers:
                op.zero_grad()
            loss.backward()
            for op in optimizers:
                op.step()           
            acc = get_acc_topk(out.cpu().data, target.cpu().data)
            acc_avg.update(acc) 

            if (idx+1) % 10 ==0:
                print_loss = 'epoch:%d, loess:%.4f'%(epoch, loss)
                for i in losses:
                    print_loss += ', %s : %.4f'%(i, eval(i))
                print(print_loss)

                logger_train.info(print_loss)
          
        print('-----time', np.array(times).mean())
        # 作用：测试并保存数据
        if (epoch+1) % 1 == 0:
            loss,acc = Validate(args, model.eval(), validateLoader)
            print('epoch:%d, acc:%.4f' % (epoch, acc))
            logger_test.info('---save model epoch:%d, acc:%.5f' % (epoch, acc[-1]))
            if acc[-1] > top_acc:
                top_acc = acc[-1]
                print('save model...')
                if top_acc_path !='':    # 删去之前的包，这里可以不适用
                    os.remove(top_acc_path)
                top_acc_path = save_model(model, args.model_name, epoch, top_acc, top=True)
            
            if (epoch+1) == args.epoch:
                top_acc_path = save_model(model, args.model_name, epoch, acc[-1], top=False)
        
def save_model(model,model_name,epoch,acc, top=True):
    checkpoints = 'experiment/checkpoints'
    fs = os.path.join(checkpoints,'%s_epoch_%d_acc_%.4f.pth'%(model_name,epoch,acc))
    torch.save(model.state_dict(),fs)
    if top: 
        torch.save(model.state_dict(), 'experiment/checkpoints/%s_top.pth'%model_name)
    return fs


def load_model(model,path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained)


def get_acc_of_out(out,target):
    choice = out.max(1)[1]
    correct = choice.eq(target.long()).sum()
    return correct.item() / float(len(target))

def get_acc_topk(out,target,topk=(1,)):
    batch_size = target.shape[0]
    topkm = max(topk)
    _, pred = out.topk(topkm, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    acc = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc.append(correct_k.mul_(1.0 / batch_size))
    return np.array(acc)


def extract_feat(args, model, path_load, path_save, data):
    pretrained = torch.load(path_load)
    model.load_state_dict(pretrained)
    ftss = []
    lass = []
    names = []
    for idx, intput_data in enumerate(tqdm(data)):
        data   = intput_data['data'].to(args.device)
        target = intput_data['target'].reshape(-1)  #.to(args.device)
        # name = intput_data['name'].reshape(-1)  #.to(args.device)
        
        with torch.no_grad():
            out,fts = model(data)
        
        ftss.append(fts.cpu().data) 
        lass.append(target.cpu().data)
        #names.append(name)
    ftss = torch.cat(ftss, dim=0).numpy()
    lass = torch.cat(lass, dim=0).numpy()
    # names = np.concatenate(names, axis=0)
    return np.save(path_save, {'fts':ftss, 'las':lass})




def Validate(args, model, validateLoader):
    # 各类准确率
    acc_avg = AverageMeter()
    loss_avg = AverageMeter()
    
    for idx, intput_data in enumerate(tqdm(validateLoader)):
        data   = intput_data['data'].to(args.device)
        target = intput_data['target'].reshape(-1)  #.to(args.device)
        
        with torch.no_grad():
            out,_ = model(data)
        
        out = out.cpu().data
        
        acc  = get_acc_topk(out, target, (1,))
        
        acc_avg.update(np.array([acc]).reshape(-1))

    return 0, acc_avg.avg

def test(model, test_loader):
    batch_correct =[]
    batch_loss =[]
    
   
    for batch_idx, (data, target) in enumerate(test_loader):
        pred,y = model(data.view(-1,784))
        loss = (pred, target)
        choice = pred.data.max(1)[1]
        correct = choice.eq(target.long()).sum()
        batch_correct.append(correct.item()/float(len(target)))
        batch_loss.append(loss.data.item())
    # print('test:',np.mean(batch_correct))
    return np.mean(batch_correct), np.mean(batch_loss)



if __name__ == '__main__':
    main()

    # img = GetDataTrain(dataType='train', imageMode='RGB', domainMode=0)

