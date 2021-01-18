import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import PIL
from torchvision import transforms
import torch
import logging
import time
from tqdm import tqdm 

_ATTR_STYLE = [
    {'id': 0, 'category': 'Modern',},
    {'id': 1, 'category': 'Chinoiserie',},
    {'id': 2, 'category': 'Kids',},
    {'id': 3, 'category': 'European',},
    {'id': 4, 'category': 'Japanese',},
    {'id': 5, 'category': 'Southeast Asia',},
    {'id': 6, 'category': 'Industrial',},
    {'id': 7, 'category': 'American Country',},
    {'id': 8, 'category': 'Vintage/Retro',},
    {'id': 9, 'category': 'Light Luxury',},
    {'id': 10, 'category': 'Mediterranean',},
    {'id': 11, 'category': 'Korean',},
    {'id': 12, 'category': 'New Chinese',},
    {'id': 13, 'category': 'Nordic',},
    {'id': 14, 'category': 'European Classic',},
    {'id': 15, 'category': 'Others',},
    {'id': 16, 'category': 'Ming Qing',},
    {'id': 17, 'category': 'Neoclassical',},
    {'id': 18, 'category': 'Minimalist',},
]


_SUPER_CATEGORIES_3D = [
    {'id': 1, 'category': 'Cabinet/Shelf/Desk'},
    {'id': 2, 'category': 'Bed'},
    {'id': 3, 'category': 'Chair'},
    {'id': 4, 'category': 'Table'},
    {'id': 5, 'category': 'Sofa'},
    {'id': 6, 'category': 'Pier/Stool'},
    {'id': 7, 'category': 'Lighting'},
]

_CATEGORIES_3D = [
    {'id': 1, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Children Cabinet'},
    {'id': 2, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Nightstand'},
    {'id': 3, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Bookcase / jewelry Armoire'},
    {'id': 4, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Wardrobe'},
    {'id': 5, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Tea Table'},
    {'id': 6, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Corner/Side Table'},
    {'id': 7, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Sideboard / Side Cabinet / Console'},
    {'id': 8, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Wine Cooler'},
    {'id': 9, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'TV Stand'},
    {'id': 10, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Drawer Chest / Corner cabinet'},
    {'id': 11, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Shelf'},
    {'id': 12, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Round End Table'},
    {'id': 13, 'category': 'Bed', 'fine-grained category name': 'Double Bed'},
    {'id': 14, 'category': 'Bed', 'fine-grained category name': 'Bunk Bed'},
    {'id': 15, 'category': 'Bed', 'fine-grained category name': 'Bed Frame'},
    {'id': 16, 'category': 'Bed', 'fine-grained category name': 'Single bed'},
    {'id': 17, 'category': 'Bed', 'fine-grained category name': 'Kids Bed'},
    {'id': 18, 'category': 'Chair', 'fine-grained category name': 'Dining Chair'},
    {'id': 19, 'category': 'Chair', 'fine-grained category name': 'Lounge Chair / Book-chair / Computer Chair'},
    {'id': 20, 'category': 'Chair', 'fine-grained category name': 'Dressing Chair'},
    {'id': 21, 'category': 'Chair', 'fine-grained category name': 'Classic Chinese Chair'},
    {'id': 22, 'category': 'Chair', 'fine-grained category name': 'Barstool'},
    {'id': 23, 'category': 'Table', 'fine-grained category name': 'Dressing Table'},
    {'id': 24, 'category': 'Table', 'fine-grained category name': 'Dining Table'},
    {'id': 25, 'category': 'Table', 'fine-grained category name': 'Desk'},
    {'id': 26, 'category': 'Sofa', 'fine-grained category name': 'Three-Seat / Multi-person sofa'},
    {'id': 27, 'category': 'Sofa', 'fine-grained category name': 'armchair'},
    {'id': 28, 'category': 'Sofa', 'fine-grained category name': 'Two-seat Sofa'},
    {'id': 29, 'category': 'Sofa', 'fine-grained category name': 'L-shaped Sofa'},
    {'id': 30, 'category': 'Sofa', 'fine-grained category name': 'Lazy Sofa'},
    {'id': 31, 'category': 'Sofa', 'fine-grained category name': 'Chaise Longue Sofa'},
    {'id': 32, 'category': 'Pier/Stool', 'fine-grained category name': 'Footstool / Sofastool / Bed End Stool / Stool'},
    {'id': 33, 'category': 'Lighting', 'fine-grained category name': 'Pendant Lamp'},
    {'id': 34, 'category': 'Lighting', 'fine-grained category name': 'Ceiling Lamp'}
]

dict_style =  {'Modern': 0, 'Chinoiserie': 1, 'Kids': 2, 'European': 3, 'Japanese': 4, 'Southeast Asia': 5, 'Industrial': 6, 'American Country': 7, 'Vintage/Retro': 8, 'Light Luxury': 9, 'Mediterranean': 10, 'Korean': 11, 'New Chinese': 12, 'Nordic': 13, 'European Classic': 14, 'Others': 15, 'Ming Qing': 16, 'Neoclassical': 17, 'Minimalist': 18}
dict_cates =  {'Cabinet/Shelf/Desk': 0, \
               'Bed': 1, \
               'Chair': 2, \
               'Table': 3, \
               'Sofa': 4, \
               'Pier/Stool': 5, \
               'Lighting': 6}
dict_subCates =  {'Children Cabinet': 0, \
                  'Nightstand': 1, \
                  'Bookcase / jewelry Armoire': 2, \
                  'Wardrobe': 3,\
                  'Tea Table': 4, \
                  'Corner/Side Table': 5, \
                  'Sideboard / Side Cabinet / Console': 6, \
                  'Wine Cooler': 7,\
                  'TV Stand': 8, \
                  'Drawer Chest / Corner cabinet': 9,\
                  'Shelf': 10, \
                  'Round End Table': 11, \
                  'Double Bed': 12, \
                  'Bunk Bed': 13,\
                  'Bed Frame': 14,\
                  'Single bed': 15, \
                  'Kids Bed': 16, \
                  'Dining Chair': 17, \
                  'Lounge Chair / Book-chair / Computer Chair': 18,\
                  'Dressing Chair': 19,\
                  'Classic Chinese Chair': 20,\
                  'Barstool': 21, \
                  'Dressing Table': 22, \
                  'Dining Table': 23, \
                  'Desk': 24, \
                  'Three-Seat / Multi-person sofa': 25,\
                  'armchair': 26, \
                  'Two-seat Sofa': 27, \
                  'L-shaped Sofa': 28, \
                  'Lazy Sofa': 29, \
                  'Chaise Longue Sofa': 30, \
                  'Footstool / Sofastool / Bed End Stool / Stool': 31, \
                  'Pendant Lamp': 32, \
                  'Ceiling Lamp': 33}

dict_cates_index =list(dict_cates.keys())
dict_subCates_index = list(dict_subCates.keys())

models_cates = [2009.,368.,  579.,  284., 1134.,  288.,  540.]
models_subCates = [ 38., 259., 156., 124., 369., 275., 259., 102., 145., 165.,
                    71.,  46., 198.,  31.,52.,  51.,  36., 173., 361.,   8.,
                    15.,  22.,  57., 167.,  60., 413., 356., 214.,88.,  53.,
                    10., 288., 405., 135.]
align_label = [#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
               [1,1,1,1,1,1,1,1,1,1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0,0,0,0,0,0,0,0,0,0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
               [0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
align = [12,17,22,25,31,34]
# model keys() 就是有又{'file nanme':[big class, mini class, style]}
path = ['model_label.json',
        'model_image.json',
        'image_label.json',
        'ttrain_test.json',
        'train_set.json']

img_path = [
    "D:\\Temp\\Data\\retrieval_train\\train\\image",
    "D:\\Temp\\Data\\retrieval_train\\train\\mask",
    "D:\\迅雷下载\\img_mask"
]

def out1_to_out0():

    '''

    function: 将out1 34个类，标签转换成out0 6个类
    :return:
    '''
    a = torch.randn(8,34)
    b = a.softmax(-1)

    align_bit = [torch.Tensor([i]).expand_as(b) for i in align_lable]
    out_0 = torch.cat([(b * i).sum(-1, keepdim=True) for i in align_bit], dim=1)
    print('b:',b,b.shape)
    print('==',out_0, out_0.shape)

    return out_0

def get_logger(model_name):
    '''
     function：可以随时写入文件中，但是每次训练的时候要更爱名字
    :return:
    '''
    log_dir = 'experiment'
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/%s.txt' % model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def image_resize():
    '''
        检验resize之后的图像怎么样？
    '''
    index = 0
    imgs = os.listdir(img_path[0])
    for idx,i in enumerate(imgs):
        img = Image.open(os.path.join(img_path[0],i))
        print(idx)
        if idx ==1000: break
        if img.height != img.width:
            index += 1
        img.close()





def img_mask(img_path):
    transform = transforms.Compose([ transforms.ToTensor()])
    imgs = sorted(os.listdir(img_path[0]))   # jpg
    masks = sorted(os.listdir(img_path[1]))  # png
    i_p = [os.path.join(img_path[0], i) for i in imgs]
    m_p = [os.path.join(img_path[1], i) for i in masks]
    for idx,(i,m,imgs) in enumerate(zip(i_p,m_p,imgs)):
        ii = Image.open(i).convert('RGB')
        mm = Image.open(m).convert('RGB')
        tmm = transform(mm)
        tii = transform(ii)
        transforms.ToPILImage()(tmm*tii).save(os.path.join(img_path[2],imgs),'JPEG')
        ii.close()
        mm.close()
        print('%d/%d'%(idx,len(i_p)))

# a0 = Image.open('m0.png')
# a0.show()
# a1 = Image.open('m0.png').convert('RGB')
# a1.show()
# a0 = transforms.ToTensor()(a0)
# a1 = transforms.ToTensor()(a1)
# for i,j,k in zip(a0[0],a1[0],a1[1]):
#     print('i:',i)
#     print('j:',j)
#     print('k:',k)
# print(a0.shape, a1.shape)




def img_transform(mode):
    if mode == 'L':
        a = Image.open(path[0]).convert('L')
        m = Image.open(path[2]).convert('L')
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    else:
        a = Image.open(path[0]).convert('RGB')
        m = Image.open(path[2]).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])







def model_max_iamge(path):
    f = open(path[1])
    model = json.load(f)
    maxs = 0
    dicts = 0
    for i in model:
        re = len(model[i])
        if re >maxs:
            maxs = re
            dicts = i
    print('max,dists:%d,%s'%(maxs, dicts))




# 作用：从model每类随机选出1/5的数据
# 思路：先将所有的index随机数选取好，需要记录每个类选择到了哪个！这种方式可以一次性获取所有的数据。
# 问题：xxx 输入的数据和输出的数据没有对上
def train_test():
    test = []
    train = []
    # 获取随机值
    count = np.zeros()
    num = np.array(models_subCates) // 5
    choice = [sorted(np.random.choice(range(int(j)), int(i), replace=False)) for i, j in
              zip(num, np.array(models_subCates) - 1)]
    # print('choice:', choice, sum([len(i) for i in choice]))
    f = open('model_label.json')
    model = json.load(f)
    #
    for i in model:
        index = model[i][1]
        # if count[index]==0:
        if count[index] in choice[index]:
            test.append(i)
            # del(choice[index][0])   # 验证是否取完
        else: train.append(i)
        count[index] += 1

    if count[index] in choice[index]:   # 最后还得补一个，但是可有可无。
        test.append(i)
    else: train.append(i)

    # json.dump({'train': train, 'test': test}, open('train_test.json', 'w'))

    print('choice:',choice)
    print('num:',num.sum(),',',len(test))

# # 得model 的统计量
# f = open('model_label.json')
# model = json.load(f)
# cates = np.zeros(7)
# subCates =np.zeros(34)
# for i in model:
#     cates[model[i][0]] +=1
#     subCates[model[i][1]] +=1
# print(cates)
# print(subCates)





#
# # 输出每个模型对应的标签，和model 与iamge的对应关系
# dict_image = {}
# dict_model = {}
# dict_modelImage = {}
# for i in train:
#     dict_image[i['image'].split('.')[0]] = [dict_cates[i['category']], dict_subCates[i['fine-grained category']],dict_style[i['style']]]
#
#     if i['model'] not in dict_model.keys():
#             dict_model[i['model']] = [dict_cates[i['category']], dict_subCates[i['fine-grained category']],dict_style[i['style']]]
#
#             dict_modelImage[i['model']] = [i['image'].split('.')[0]]
#     else: dict_modelImage[i['model']].append(i['image'].split('.')[0])
#
# json.dump(dict_image, open('image_label.json','w'))
# json.dump(dict_model, open('model_label.json','w'))
# json.dump(dict_modelImage, open('model_image.json','w'))



# function:将obj 中，g和o 都注释掉
# path = 'D:/Temp/python/data/retrieval_train/model/'
# # ===============train 中的model
# pathr = 'D:\\Temp\\Python\\3D\\data\\retrieval_train\\train\\model'
# pathw = 'D:\\Temp\\Python\\3D\\data\\retrieval_train\\train\\model1'
# ===============validation 的数据
# pathr = 'D:\\Temp\\Python\\3D\\data\\retrieval_validation\\validation\\model'
# pathw = 'D:\\Temp\\Python\\3D\\data\\retrieval_validation\\validation\\model1'
#
# # ========================================
# # fsr = os.listdir(pathr)
# for k,file in enumerate(fsr):
#     fr = open(os.path.join(pathr,file), 'r')
#     fw = open(os.path.join(pathw,file), 'w')
#     fr_lines = fr.readlines()
#     for i in range(len(fr_lines)):
#         if fr_lines[i][0:2] == 'o ' or fr_lines[i][0:2] == 'g ':
#             fr_lines[i] = '# ' + fr_lines[i]
#     fw.writelines(fr_lines)
#     fw.close()
#     fr.close()
#     print(k)

# ========================================
# 检测是否干净了
# fsr = os.listdir(pathw)
# count = 0
# name = []
# for k,file in enumerate(fsr):
#     fr = open(os.path.join(pathw,file), 'r')
#     fr_lines = fr.readlines()
#     for i in range(len(fr_lines)):
#         if fr_lines[i][0:2] == 'g ':
#             name.append(file)
#             count +=1
#             break
#     fr.close()
#     if count ==10:break
#     print('k',k)
# print('name:', name,',',len(name))





# # 验证3D数据的数字并不是连续的
# pathr = 'D:\\Temp\\Python\\3D\\data\\retrieval_train\\train\\image'
# fs = os.listdir(pathr);
# # print(fs)
# fs = [int(i.split('.')[0]) for i in fs]
# c=0
# count=0
# for i in fs:
#
#     if c!=i:
#         print('c,i:',c,',',i)
#         c=i
#         count+=1
#     c += 1
#
# print(count)
def Transff(mode):
    if mode =='L':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    else:#'RGB'
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


def processData():
    pathDomain = ["/home/dh/zdd/data/train_retrieval/image",\
                   "/home/dh/zdd/data/train_retrieval/image_render5",\
                   "/home/dh/zdd/data/train_retrieval/image_mask"\
                    ]
#    pathNew = ["/home/dh/zdd/data/pth/image",\
#                   "/home/dh/zdd/data/pth/render5",\
#                   "/home/dh/zdd/data/pth/mask"\
#                    ]
    pathNew = ["/home/dh/zdd/data/pth_gray/image",\
                   "/home/dh/zdd/data/pth_gray/render5",\
                   "/home/dh/zdd/data/pth_gray/mask"\
                    ]
    
    index = 0
    imd = 'L' #'RGB'
    imgs = os.listdir(pathDomain[index])
    transf = Transff(imd)
    
    for img_name in tqdm(imgs):
        path = os.path.join(pathDomain[index],img_name)
        path_new = os.path.join(pathNew[index], img_name.split('.')[0] + '.txt')
        img = Image.open(path).convert(imd)
        data = transf(img).squeeze().numpy()
        np.savetxt(path_new,data,fmt='%.6f') 




if __name__ == '__main__':
    # img_mask(img_path)
    # image_resize()
    # logger = get_logger('train_ali')
    processData()
   
    
    # 看内存能够放多
