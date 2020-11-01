import random
from random import choice
import ruleEngine
import os
import detection
import img_multi_label

# model, inp, transform_com = img_multi_label.init_model()
# for i in range(100):
#
#     ans = img_multi_label.seach_label('G:\\ML-GCN-master\\data\\train2014\\COCO_train2014_000000000034.jpg', model, inp, transform_com)
#define hyper parameter
K=3
K_POINT=2
COUNT=5
NUMBER=11
MAX_ITER=20
CROSS_RATE=0.75
MUTATION_RATE=0.1
TYPELOSS=[10,5,1]
INITIAL_LIST=[0,1,4,6,7]
OFF_POOL=[0,1,2,3,4,5,6,7,8,9,10]
BASE_PATH='.\\GA_music'
#intialize population
def initialize_pop():
    population=[]
    for k in range(COUNT):
        population.append([INITIAL_LIST[k]]*10)
    return population

#evaluate fitness value
type_dict={'passion1':1,'passion-1':2, 'excited1':3, 'excited-1':4, 'happy1':5, 
           'happy-1':6, 'relaxed-1':7, 'relaxed1':8, 'quiet-1':9, 'quiet1':10}
#fitness value: 相反大类，diff*10,不同大类，diff*5,小类，diff*1 

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def parse_label(output):
    if is_number(output[-3]):
        change = int(output[-2:])
        key = output[:-2]
        print(key,change)
    else:
        change = int(output[-1])
        key = output[:-1]
    music_type = abs(type_dict[key] - change)
    return music_type

def parse_folder(filename):
    return type_dict[filename]

def evaluate_ind(true_label,output_label):
    diff=abs(true_label-output_label)
    if diff>5:
        fit_value=diff*TYPELOSS[0]
    elif diff>1:
        fit_value=diff*TYPELOSS[1]
    else:
        fit_value=diff*TYPELOSS[2]
    return fit_value

def evaluate_pop(true_label,output_labels):
    fit_values=[]
    for c in output_labels:
        fit_values.append(evaluate_ind(true_label,c))
    return fit_values

#evaluate if stopping criteria is satisfied
def evaluate_stop(fit_values,iter):
    return min(fit_values)<=0 or iter>MAX_ITER

#chromosome selection，using k-tournament (K=3)selection method
def tour_selection(population,fit_values):
    parents=[]
    for _ in range(COUNT):
        min_fit=100
        index=-1
        for k in range(K):
            i=random.randint(0,4)
            if min_fit>fit_values[i]:
                min_fit=fit_values[i]
                index=i
        parents.append(population[index])
    return parents
#染色体交叉，多点交叉，K_POINT=2
#chromosome crossover using multi-point method
def cross(p1,p2):
    indexs=[]
    for _ in range(K_POINT):
        indexs.append(random.randint(0,NUMBER-1))
    index1=max(indexs)
    index2=min(indexs)
    temp_list=p1[index2:index1+1]
    p1[index2:index1+1]=p2[index2:index1+1]
    p2[index2:index1+1]=temp_list
    return p1,p2
def multi_point_crossover(parents):
    for i in range(COUNT-1):
        r=random.random()
        if r<CROSS_RATE:
            cross(parents[i],parents[i+1])
    return parents

#染色体突变，位翻转方式
#chromosome mutation, using bit flip
def mutate(offspring):
    for i in range(COUNT):
        for j in range(NUMBER-1):
            r=random.random()
            if r<MUTATION_RATE:
                old_value=offspring[i][j]
                # print(OFF_POOL,old_value)
                OFF_POOL.remove(old_value)
                new_value=choice(OFF_POOL)
                offspring[i][j]=new_value
                OFF_POOL.append(old_value)
    return offspring

def write_to_txt(filename, population):
    with open(filename, 'w') as f:
        res = population[0]
        for i in res:
            f.write(str(i))
    f.close()
    
#train GA using videos
def train():
    population=initialize_pop()
    print('initial population:',population)
    iter=0
    output_labels = []
    for m_type in os.listdir(BASE_PATH):
        type_path=BASE_PATH+'/'+m_type
        for degree in os.listdir(type_path):
            filepath=type_path+'/'+degree
            print(degree)
            true_label=parse_folder(degree)

            for videoname in os.listdir(filepath):
                output_labels = []
                upload=filepath+'/'+videoname
                image_path = detection.extract_frame(upload)
                for c in population:
                    output=ruleEngine.find_image_label(image_path,c)
                    if not is_number(output[-2]):
                        break
                    print(output)
                    output_labels.append(parse_label(output))
                print('output labels', output_labels)
                if output_labels==[]:
                    print('no output label!')
                    continue
                fit_values=evaluate_pop(true_label,output_labels)
                print('fitness value:', fit_values)
                if evaluate_stop(fit_values,iter):
                    return population
                parents=tour_selection(population,fit_values)
                offspring=multi_point_crossover(parents)
                population=mutate(offspring)
                print(f"{iter} iteration:",population)
                iter+=1
    return population

res_pop=train()
write_to_txt('chromosome.txt', res_pop)
