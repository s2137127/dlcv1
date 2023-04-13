import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.decomposition import PCA
from data import Bike
from sklearn import manifold
from model import MyNet
# python draw.py ../hw1_data/p1_data
feature_dic = {}
def get_activation(name):
    def hook(model, input, output):
        feature_dic[name] = output.detach()

    return hook

if __name__ == "__main__":
    data_path = sys.argv[1]
    # model = models.vgg19_bn(pretrained=False)
    # model.classifier[6].out_features = 50
    model = MyNet()
    model.load_state_dict(torch.load('./081748.pth'))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(size=(112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    val_set = Bike(root_dir=os.path.join(data_path, "val_50"), transform=transform)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    print('Length of val Set:', len(val_set))
    label_arr = []
    out_arr = []
    # print(model)
    with torch.no_grad():
        handle = model.fc[2].register_forward_hook(get_activation('features'))
        # handle = model.classifier[3].register_forward_hook(get_activation('features'))
        for _, (x, label) in enumerate(val_loader):
            if use_cuda:
                x = x.cuda()

            y = model(x)
            label_arr.append(label)
            out_arr.append(feature_dic['features'].to("cpu").numpy()[0])

    handle.remove()
    feature = np.array(out_arr.copy())
    mean_vector = np.array(out_arr.copy()).mean(axis=0)  # 計算縱行算術平均數
    print(mean_vector.shape)
    pca = PCA(n_components = 2)
    X = pca.fit_transform(out_arr - mean_vector)

    # print("X shape:", X.components_.shape)
    print("pca finish")
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=0).fit_transform(feature)
    # print("tsne:", X_tsne.shape)
    # Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
    print(X.shape)
    print(X_norm.shape)
    print(label_arr[2].item())
    fig = plt.figure(figsize=(16,8 ))

    ax1 = fig.add_subplot(121)
    ax1.set_title("PCA")
    ax1.scatter(X[:,0], X[:,1], c=label_arr)
    for i in range(X.shape[0]):
        ax1.annotate(str(label_arr[i].item()),xy = (X[i,0],X[i,1]),xytext = (X[i,0],X[i,1]),alpha = 0.1,annotation_clip = True)
    ax2 = fig.add_subplot(122)
    ax2.set_title("TSNE")
    ax2.scatter(X_norm[:, 0], X_norm[:, 1], c=label_arr , cmap = 'rainbow')
    for i in range(X.shape[0]):
        ax2.annotate(str(label_arr[i].item()),xy = (X_norm[i,0],X_norm[i,1]),xytext = (X_norm[i,0],X_norm[i,1]),alpha = 0.1,annotation_clip = True)
    plt.savefig('data_visdualize.jpg')

    plt.show()


