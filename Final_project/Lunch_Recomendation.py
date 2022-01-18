from operator import index
from re import U
from select import select
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#-------read files-------#
train = pd.read_csv('20220113_oht.csv')
#test_x = pd.read_csv('test.csv')
target_train = pd.read_csv('target.csv' , header = None,)
#target_test = pd.read_csv('test_sol.csv' , header = None )

df_train_list = train.values.tolist()
#print(df_train_list[315])
#df_test_list = test_x.values.tolist()
target_train = target_train.values.tolist()
target_train = [(idx, item) for idx,item in enumerate(target_train)]
#print(target_train)
# target_test = target_test.values.tolist()
# target_test = [(idx, item) for idx,item in enumerate(target_test)]

# img = mpimg.imread('region.png')
# imgplot = plt.imshow(img)
# plt.show()

def cosine_sim(vec_a, vec_b):
    dot = sum(a*b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a*a for a in vec_a) ** 0.5
    norm_b = sum(b*b for b in vec_b) ** 0.5
    cos_sim = dot / (norm_a*norm_b)
    return cos_sim


def hamming_distance(vec_a, vec_b):
	dist_counter = 0
	for n in range(len(vec_a)):
		if vec_a[n] != vec_b[n]:
			dist_counter += 1
	return dist_counter  


def find_same_vec(df_train_list):
    vec_all = []
    for i in range(len(df_train_list)):
        vec_idx = []
        for j in range(len(df_train_list)):
            if i+j > 346 :
                pass
            elif df_train_list[i+j][1:] == df_train_list[i][1:]:
                vec_idx.append(df_train_list[i+j][0])
            else:
                pass       
        vec_all.append(vec_idx)  
    return vec_all

def find_mul_vec(vec_all):
    vec_mul = []
    for i in range(len(vec_all)):
        if len(vec_all[i]) > 2:
            vec_mul.append(vec_all[i])
    return vec_mul


def give_target(vec_mul, target_train):
    for i in range(len(vec_mul)):
        temp = []
        for j in range(len(vec_mul[i])):
            temp.append(target_train[vec_mul[i][j]][1]) 
        temp1 = []
        for item in temp:
            temp1 += item
        label = max(temp1, key=temp1.count)
        lab = []
        lab.append(label)
        vec_mul[i] = vec_mul[i] + lab
    #print(vec_mul)
    return vec_mul

def compare(vec_mul, vec_a):
    CS_list = []
    HD_list = []
    for i in range(len(vec_mul)):
        vec_b = df_train_list[vec_mul[i][0]][1:]
        CS_list.append(cosine_sim(vec_a, vec_b))
        HD_list.append(hamming_distance(vec_a, vec_b))
    Choose1 = max(CS_list)
    choose2 = min(HD_list)
    print("Cosine similarity =  ", Choose1)
    print("Hamming distance = ", choose2)
    index1 = CS_list.index(Choose1)
    index2 = HD_list.index(choose2)
    print("input = ", vec_a)
    print("reference_cos = " , df_train_list[vec_mul[index1][0]][1:])
    print("reference_ham = " , df_train_list[vec_mul[index2][0]][1:])
    final1 = vec_mul[index1][-1]
    final2 = vec_mul[index2][-1]
    food_dic = {"水木自助餐" : 0,
    "校內便利商店": 1,
    "風雲樓二三樓": 2,
    "校外學校附近（清夜）": 3,
    "清大小吃部" : 4,
    "麥當勞" : 5,
    "外送（Uber Eats, Foodpanda)" : 6}
    V = list(food_dic.values())
    K = list(food_dic.keys())
    for i,j in zip(V,range(len(K))):
        if final1 == V[i]:
            print("The system recommend you  select (base on cosine similarity) :", K[j] )
    for i,j in zip(V,range(len(K))):
        if final2 == V[i]:
            print("The system recommend you  select (base on hamming distance):", K[j] )
    
        
def evaluate(final_list, target_test):
    true = 0
    false = 0
    for i,j in zip(final_list, target_test):
        if i == j:
            true += 1
        else:
            false += 1
        accuracy = true / true + false
    return accuracy

def Ques1():
    b = 'boy'
    g = 'girl'
    print(("Boy or Girl : " ))
    gender = str(input())
    if gender == b :
        vec_a.extend([1, 0, ])
    elif gender == g :
        vec_a.extend([0, 1, ])
    else:
        print("Type wrong")
        return(Ques1())        

def Ques2():  
    u  = 'under graduate'
    m = 'master'
    p = 'phd'
    pf = 'professor'
    s = 'staff'
    print("Identity(Under Graduate, Master, PHD, Professor, staff) : ")
    identity = str(input())
    if identity == u :
        vec_a.extend([1, 0, 0, 0, ])
    elif identity == m :
        vec_a.extend([0, 1, 0, 0, ])
    elif identity == p :
        vec_a.extend([0, 0, 1, 0, ])
    elif identity == s or identity == pf :
        vec_a.extend([0, 0, 0, 1, ])
    else:
        print("Type wrong")
        return(Ques2())

def Ques3():
    u = 'less'
    h = 'more'
    print("Study less or more than 2 years")
    study = str(input())
    if study == u :
        vec_a.extend([1, 0, ])
    elif study == h :
        vec_a.extend([0, 1, ])
    else:
        print("Type wrong")
        return(Ques3())

def Ques4():
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'
    print("Where's your position : ")
    region = str(input())
    if region == a :
        vec_a.extend([1, 0, 0, 0, 0, ])
    elif region == b :
        vec_a.extend([0, 1, 0, 0, 0, ])
    elif region == c :
        vec_a.extend([0, 0, 1, 0, 0, ])
    elif region == d :
        vec_a.extend([0, 0, 0, 1, 0, ])
    elif region == e :
        vec_a.extend([0, 0, 0, 0, 1, ])
    else:
        print("Type wrong")
        return(Ques4())

def Ques5():
    y = 'yes'
    n = 'no'
    print("Have you eat breakfast (Yes/No) : ")
    breakfast = str(input())
    if breakfast == y :
        vec_a.extend([1, 0, ])
    elif breakfast == n :
        vec_a.extend([0, 1, ])
    else:
        print("Type wrong")
        return(Ques5())

def Ques6():
    y = 'yes'
    n = 'no'
    print("Are you on the diet (Yes/No) : ")
    diet = str(input())
    if diet == y :
        vec_a.extend([1, 0, ])
    elif diet == n :
        vec_a.extend([0, 1, ])
    else:
        print("Type wrong")
        return(Ques6())

def Ques7():
    a = '0~50'
    b = '50~100'
    c = '100~200'
    d = '200up'
    print("What's your lunch budget (0~50 or 50~100 or 100~200 or 200up) : ")
    budget = str(input())
    if budget == a :
        vec_a.extend([1, 0, 0, 0, ])
    elif budget == b :
        vec_a.extend([0, 1, 0, 0, ])
    elif budget == c :
        vec_a.extend([0, 0, 1, 0, ])
    elif budget == d :
        vec_a.extend([0, 0, 0, 1, ])
    else:
        print("Type wrong")
        return(Ques7())

def Ques8():
    s = 'sunny'
    c = 'cloudy'
    r = 'rainy'
    cd ='rain cat and dog'
    print("What's today's weather(sunny or cloudy or rainy or rain cat and dog) : ")
    weather = str(input())
    if weather == s :
        vec_a.extend([1, 0, 0, 0, ])
    elif weather == c :
        vec_a.extend([0, 1, 0, 0, ])
    elif weather == r :
        vec_a.extend([0, 0, 1, 0, ])
    elif weather == cd :
        vec_a.extend([0, 0, 0, 1, ])
    else:
        print("Type wrong")
        return(Ques8())

def Ques9():
    h = 'hot'
    o = 'common'
    c = 'cold'
    print("What's today;s temparature(Hot or Common or Cold) : ")
    temparature = str(input())
    if temparature == h :
        vec_a.extend([1, 0, 0])
    elif temparature == o :
        vec_a.extend([0, 1, 0])
    elif temparature == c :
        vec_a.extend([0, 0, 1])
    else:
        print("Type wrong")
        return(Ques9())
        

if __name__ == '__main__':
    # vec_a = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    vec_a = []
    # for i in range(0, 28):
    #     print("Enter the attribute:" , i)
    #     item = int(input())
    #     vec_a.append(item)
    print("Please type all lowercase letter")
    Ques1()
    Ques2()
    Ques3()
    img = mpimg.imread('region.png')
    imgplot = plt.imshow(img)
    plt.show()
    Ques4()
    Ques5()
    Ques6()
    Ques7()
    Ques8()
    Ques9()
    vec_mul = find_mul_vec(find_same_vec(df_train_list))
    input = give_target(vec_mul, target_train)
    compare(input, vec_a)
    












