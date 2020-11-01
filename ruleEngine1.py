import random

#read txt method three
# nums0 = []
# nums1 = ['bicycle']
# nums2 = ['bicycle', 'cat']

import detection
import img_multi_label as img_read
import cv2

def find_music_maintype(num):
    musicType = -1
    f2 = open("rules.txt","r")
    lines = f2.readlines()
    labelLen = len(num)
    # print('labelLen',labelLen)

    for line in lines:

        # first step to confirm music type
        number = filter(str.isdigit, line)
        number = list(number)
        number = int(number[0]) if len(number) > 0 else None

        if number == labelLen:
            selType = line.split('=')
            selType = str(selType[-1]).strip('\n')
            #print(selType)
            #print('one')
            if selType == 'random':
                musicType = 0
            if selType == 'one':
                musicType = 1
            if selType == 'two':
                musicType = 2
            #print(musicType)
    f2.close()
    return musicType




def find_music_subtype(num):
    f2 = open("rules.txt", "r")
    lines = f2.readlines()
    count = 0
    for line in lines:
        count += 1
        if 11 <= count <= 15:

            line = line.split('=')
            condition = line[-1].strip('\n')
            line = line[2].split('then')
            # print(line[0],num)
            # print('animal','animal')
            if num == line[0].strip(' '):
                return condition

def find_music_type(num):
    f2 = open("rules.txt", "r")
    lines = f2.readlines()
    count = 0
    for line in lines:
        count += 1
        if 5 <= count <= 9:
            line = line.split('=')
            condition = line[-1].strip('\n')
            line = line[3].split('then')
            # print(line)

            if num == line[0].strip(' '):
                return condition

dicts = {
    'baseball bat': 'sport', 'baseball glove': 'sport', 'frisbee': 'sport', 'kite': 'sport', 'remote': 'sport',
    'snowboard'
    : 'sport', 'sports ball': 'sport', 'surfboard': 'sport', 'tennis racket': 'sport', 'boat':'sport', 'skis':'sport',

    'airplane': 'car', 'bicycle': 'car', 'fire hydrant': 'car', 'motorcycle': 'car', 'parking meter': 'car',
    'stop sign': 'car','sink':'car','skateboard':'car'
    , 'traffic light': 'car', 'train': 'car', 'truck': 'car', 'car': 'car', 'bus': 'car',

    'bear': 'animal', 'carrot': 'animal', 'cat': 'animal', 'cow': 'animal', 'dog': 'animal', 'elephant': 'animal',
    'giraffe': 'animal', 'sheep':'animal',
    'horse': 'animal', 'mouse': 'animal', 'person': 'animal', 'teddy bear': 'animal', 'zebra': 'animal',
    'bird': 'animal',

    'backpack': 'furniture', 'bench': 'furniture', 'cell phone': 'furniture', 'chair': 'furniture',
    'clock': 'furniture', 'couch': 'furniture',
    'cup': 'furniture', 'dining table': 'furniture', 'bowl': 'furniture', 'bed': 'furniture',
    'fork': 'furniture', 'hair drier': 'furniture', 'handbag': 'furniture', 'keyboard': 'furniture',
    'knife': 'furniture', 'laptop': 'furniture','scissors':'furniture'
    , 'microwave': 'furniture', 'potted plant': 'furniture', 'refrigerator': 'furniture', 'suitcase': 'furniture',
    'bottle': 'furniture',
    'tie': 'furniture', 'toilet': 'furniture', 'toothbrush': 'furniture', 'tv': 'furniture', 'umbrella': 'furniture',
    'vase': 'furniture', 'wine glass': 'furniture', 'book':'furniture',

    'apple': 'eat', 'banana': 'eat', 'broccoli': 'eat', 'donut': 'eat', 'hot dog': 'eat', 'orange': 'eat',
    'oven': 'eat', 'pizza': 'eat','sandwich':'eat'
    , 'cake':'eat','spoon': 'eat', 'toaster': 'eat'}

Array = [-1,1,-1,1,-1,1,-1,1,-1,1]
change_array=[]
dict_music = {'passion_light':Array[0],'passion_high':Array[1], 'quiet_light':Array[2],'quiet_high':Array[3],
              'relaxed_light':Array[4],'relaxed_high':Array[5],'happy_light':Array[6],'happy_high':Array[7],
              'excited_light':Array[8],'excited_high':Array[9]}


# lens = len(nums2)
#
# print(find_music_maintype(nums2))
# print(find_music_subtype(dicts[nums2[1]]))
# print(find_music_type(dicts[nums2[0]]))

def read_txt(filename):
    with open(filename, 'r') as f:
        for line in f:
            for i in range(len(line)):
                change_array.append(int(line[i]))
    f.close()
    dict_change = {'passion_high': change_array[0], 'passion_light': change_array[1], 'excited_high': change_array[2],
                       'excited_light': change_array[3],
                       'happy_high': change_array[4], 'happy_light': change_array[5], 'relaxed_light': change_array[6],
                       'relaxed_high': change_array[7],
                       'quiet_light': change_array[8], 'quiet_high': change_array[9]}
    return dict_change

def find_music(nums2):
    music_maintype = find_music_maintype(nums2)
    if music_maintype == 0:
        print('random select one')
        randomNum = random.randint(1, 5)
        dicc = {1:'passion',2:'quiet',3:'happy',4:'relaxed',5:'excited'}
        music_type = dicc[randomNum]
        music_type = music_type + str(0)


    if music_maintype == 1:
        # print('play one music')
        music_type = find_music_type(dicts[nums2[0]])
        music_type = music_type.split('_')
        music_type = music_type[0]+str(0)



    if music_maintype == 2:
        print('play one specific music')

        music_type = find_music_type(dicts[nums2[0]])
        music_subtype = find_music_subtype(dicts[nums2[1]])

        print('the two labels',dicts[nums2[0]], dicts[nums2[1]])

        f2 = open("rules.txt", "r")
        lines = f2.readlines()
        count = 0
        for line in lines:
            count += 1
            if 21 <= count <= 30:
                line = line.split('=')
                # print(line)
                music_type_dic = line[1].split('and')
                music_type_dic = music_type_dic[0].strip(' ')
                music_subtype_dic = line[2].split('then')
                music_subtype_dic = music_subtype_dic[0].strip(' ')
                print(music_type_dic,music_type,music_subtype_dic,music_subtype)
                if music_type_dic == music_type and music_subtype_dic == music_subtype:
                    print('keys error')
                    keys = line[-1].strip('\n')
        dict_change=read_txt('chromosome.txt')
        conditions = dict_music[keys]
        changes = dict_change[keys]
        # print(conditions)
        music_type = music_type.split('_')
        # print(music_type[0]+str(conditions))
        print('music_type',music_type,conditions,changes)
        music_type = music_type[0]+str(conditions)+str(changes)
    return music_type
