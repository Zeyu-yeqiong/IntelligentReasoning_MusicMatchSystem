import random
from random import choice
import ruleEngine
import os

#define hyper parameter
K=3
K_POINT=2
COUNT=5
NUMBER=11
MAX_ITER=20
CROSS_RATE=0.75
MUTATION_RATE=0.1
TYPELOSS=[5,3,1]
INITIAL_LIST=[0,1,4,6,7]
OFF_POOL=[0,1,2,3,4,5,6,7,8,9,10]
BASE_PATH='G:\\ML-GCN-master\\GA_music'
#intialize population
def initialize_pop():
    population=[]
    for k in range(COUNT):
        population.append(INITIAL_LIST[k]*10)
    return population

#evaluate fitness value
type_dict={'passion1':1,'passion-1':2, 'excited1':3, 'excited-1':4, 'happy1':5, 
           'happy-1':6, 'relax-1':7, 'relax1':8, 'quiet-1':9, 'quiet1':10}
#fitness value: 相反大类，diff*10,不同大类，diff*5,小类，diff*1 

def parse_label(output):
    change=int(output[-1])
    key=output[:-1]
    music_type=abs(type_dict[key]-change)
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

#验证是否停止
def evaluate_stop(fit_values,iter):
    return min(fit_values)<10 and iter<MAX_ITER

#染色体选择，k-tournament selection method
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
def mutate(offspring):
    for i in range(COUNT):
        for j in range(NUMBER):
            r=random.random()
            if r<MUTATION_RATE:
                old_value=offspring[i][j]
                print(OFF_POOL,old_value)
                OFF_POOL.remove(old_value)
                new_value=choice(OFF_POOL)
                offspring[i][j]=new_value
                OFF_POOL.append(old_value)
    return offspring

def train():
    population=initialize_pop()
    iter=0
    for m_type in os.listdir(BASE_PATH):
        type_path=BASE_PATH+'/'+m_type
        for degree in os.listdir(type_path):
            filepath=type_path+'/'+degree
            print(degree)
            true_label=parse_folder(degree)
            #rule engine
            output_labels=[]
            for c in population:
                print(filepath)
                output=ruleEngine.find_image_label(filepath,c)
                output_labels.append(parse_label(output))
            fit_values=evaluate_pop(true_label,output_labels)
            if evaluate_stop(fit_values,iter):
                return population
            parents=tour_selection(population,fit_values)
            offspring=multi_point_crossover(parents)
            population=mutate(offspring)
            iter+=1
    return population

print(train())