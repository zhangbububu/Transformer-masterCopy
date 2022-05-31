import torch

from sklearn.decomposition import PCA
from test import init, eva_one_sentence
from setting import TRAIN_FILE, DEV_FILE
from train import model
from data_pre import PrepareData
import matplotlib.pyplot as plt
model.load_state_dict(torch.load('save/model_100ep.pt', map_location=torch.device('cpu')))
model.eval().to('cpu')


x = torch.tensor([8 ,
5,
9,



823,
267,
                480,
                  196],requires_grad=False).long()

embed = model.src_embed[0]


x_em = embed(x).detach().numpy()
print(embed(x).shape)
pca=PCA(n_components=2)  #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x_em)#对样本进行降维
print(x,reduced_x)
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]


for id,i in enumerate(reduced_x):
 print(id,i)
 if id<=2:
  red_x.append(i[0])
  red_y.append(i[1])

 elif id <= 4:
  blue_x.append(i[0])
  blue_y.append(i[1])

 else:
  green_x.append(i[0])
  green_y.append(i[1])

plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
