import argparse
from engine import *
from models import *
from coco import *
from util import *
import torchvision.transforms as transforms
import cv2

def init_model():
    num_classes = 80
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/coco/coco_adj.pkl')
    checkpoint = torch.load('coco_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])


    with open('data/coco/coco_glove_word2vec.pkl', 'rb') as f:
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
    return model, inp, transform_com
def seach_label(root, model, inp, transform_com):
    img = cv2.imread(root)
    img = cv2.resize(img, (448, 448))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    # print(img,img.shape)
    # img = transform_com(img)




    img_inp = np.ones((1, 3, 448, 448))
    inp_inp = np.ones((1, 80, 300))
    inp_inp[0] = inp
    inp = inp_inp
    img_inp[0][0] = img[:,:,0]
    img_inp[0][1] = img[:,:,1]
    img_inp[0][2] = img[:,:,2]
    # print(img_inp)
    inp = torch.from_numpy(inp).float()
    img_inp = torch.from_numpy(img_inp).float()
    # print(img_inp, type(img_inp), img_inp.shape)
    # print(inp, type(inp), inp.shape)
    # print(img_inp)
    output = model(img_inp, inp)
    # print(output.shape)
    cat2idx = {'airplane': 0, 'apple': 1, 'backpack': 2, 'banana': 3, 'baseball bat': 4, 'baseball glove': 5,
                       'bear': 6, 'bed': 7, 'bench': 8, 'bicycle': 9, 'bird': 10, 'boat': 11, 'book': 12, 'bottle': 13,
                       'bowl': 14, 'broccoli': 15, 'bus': 16, 'cake': 17, 'car': 18, 'carrot': 19, 'cat': 20,
                       'cell phone': 21, 'chair': 22, 'clock': 23, 'couch': 24, 'cow': 25, 'cup': 26, 'dining table': 27,
                       'dog': 28, 'donut': 29, 'elephant': 30, 'fire hydrant': 31, 'fork': 32, 'frisbee': 33, 'giraffe': 34,
                       'hair drier': 35, 'handbag': 36, 'horse': 37, 'hot dog': 38, 'keyboard': 39, 'kite': 40, 'knife': 41,
                       'laptop': 42, 'microwave': 43, 'motorcycle': 44, 'mouse': 45, 'orange': 46, 'oven': 47,
                       'parking meter': 48, 'person': 49, 'pizza': 50, 'potted plant': 51, 'refrigerator': 52, 'remote': 53,
                       'sandwich': 54, 'scissors': 55, 'sheep': 56, 'sink': 57, 'skateboard': 58, 'skis': 59,
                       'snowboard': 60, 'spoon': 61, 'sports ball': 62, 'stop sign': 63, 'suitcase': 64, 'surfboard': 65,
                       'teddy bear': 66, 'tennis racket': 67, 'tie': 68, 'toaster': 69, 'toilet': 70, 'toothbrush': 71,
                       'traffic light': 72, 'train': 73, 'truck': 74, 'tv': 75, 'umbrella': 76, 'vase': 77,
                       'wine glass': 78, 'zebra': 79}
    ans = []
    ans_possi = []
    cat_real = {value: key for key, value in cat2idx.items()}
    for i in range(len(output[0])):
        if output[0][i] > 0:
            ans.append(cat_real.get(i))
            ans_possi.append(output[0][i])

    for i in range(len(ans_possi)):
        ans_possi[i] = float(ans_possi[i])
    dict = {'A': ans, 'B': ans_possi}
    frame = pd.DataFrame(dict)
    # print(ans_possi)
    # print(frame)
    frame = frame.sort_values('B')
    # print(frame)
    ans = frame['A'].values
    # print(ans[::-1])
    ans = ans[::-1]
    ans = ans[:2]
    # print(output,ans)

    return ans
#
# model, inp, transform_com = init_model()
# ans = seach_label('G:\ML-GCN-master\project\\app\static\\images\\20201017233004\\images\\test.jpg', model, inp, transform_com)
# print(ans)