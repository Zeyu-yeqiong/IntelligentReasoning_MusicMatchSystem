import argparse
from engine import *
from models import *
from coco import *
from util import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', default='G:\\ML-GCN-master\\',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='../../coco_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',default=True, action='store_true',
                    help='evaluate model on validation set')


def main_coco():
    global args, best_prec1, use_gpu, use_cpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    use_cpu = torch.device("cpu")

    # train_dataset = COCO2014(args.data, phase='train', inp_name='data/coco/coco_glove_word2vec.pkl')
    print(os.getcwd())
    val_dataset = COCO2014(args.data, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    num_classes = 80

    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/coco/coco_adj.pkl')
    checkpoint = torch.load('coco_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    img = Image.open('G:\\ML-GCN-master\\project\\app\\static\\images\\test.jpg').convert('RGB')

    with open('data\\coco\\coco_glove_word2vec.pkl', 'rb') as f:
        inp = pickle.load(f)
    # img = np.array(img)
    # img = transform(img)
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                     std=model.image_normalization_std)
    transform_com = transforms.Compose([
        MultiScaleCrop(448, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset.transform = transform_com
    img = transform_com(img)
    img_inp = np.ones((1,3,448,448))
    inp_inp = np.ones((1, 80, 300))
    inp_inp[0] = inp
    inp = inp_inp
    img_inp[0] = img
    inp = torch.from_numpy(inp).float()
    img_inp = torch.from_numpy(img_inp).float()
    print(img_inp, type(img_inp),img_inp.shape)
    print(inp, type(inp),inp.shape)

    output = model(img_inp,inp)
    print(output.shape)
    exit()

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/coco/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr

    state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)




    # engine.learning(model, criterion, train_dataset, val_dataset, optimizer)
    engine.learning(model, criterion, val_dataset, optimizer)
if __name__ == '__main__':
    main_coco()
