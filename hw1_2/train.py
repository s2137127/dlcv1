import datetime
import os
import random

import imageio.v2
import torch
import numpy as np
from data import get_dataloaders
from model import FCN32 ,UNet
from tqdm import tqdm
import time
import torch.nn.functional as F
from mean_iou_evaluate import mean_iou_score
config = {
    'epochs': 30,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'LEARNING_RATE': 1e-3,
    'Weight_Decay': 0.0001,
    # 'pretrained': '213055.pth',
    'pretrained': None,
    'criterion': torch.nn.CrossEntropyLoss(),
    'cnt':0,
    'best_acc':0,
    'timestr':time.strftime("%H%M%S")
}




def trainer(model, tr_loader,val_loader):
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=config['LEARNING_RATE'],
    #                              weight_decay=config['Weight_Decay'])
    # optimizer = torch.optim.Adadelta(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(),lr = config['LEARNING_RATE'],momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train_acc = 0
    train_loss = 0
    val_acc = 0
    val_loss = []
    model.train()

    for n_batch, sample_batched in enumerate(tqdm(tr_loader),1):
        optimizer.zero_grad()

        torch.cuda.empty_cache()
        data, target = sample_batched
        data, target = data.to(config['DEVICE']), target.to(config['DEVICE'])

        score = model(data)
        loss = criterion(score, target)
        loss_data = loss.item()
        train_loss += loss_data

        output = torch.argmax(F.softmax(score, dim=1), dim=1)
        correct = torch.eq(output, target).int()
        accuracy = float(correct.sum()) / float(correct.numel())

        # acc = (torch.max(score.cpu(), 1)[1] == target.cpu()).float().mean()
        # print(acc)
        train_acc += accuracy
        # show acc loss
        if n_batch % 500 == 0 or n_batch == len(tr_loader):
            print("train_loss: ", train_loss/n_batch,"train_acc: ",train_acc/n_batch)
            # loss_arr.append(loss_data)
            # acc_arr.append(acc)
            # mean_iou_score(lbl_pred,target)
        loss.backward()
        optimizer.step()
    model.eval()
    for n_batch, sample_batched in enumerate(val_loader, 1):
        torch.cuda.empty_cache()
        data, target = sample_batched
        data, target = data.to(config['DEVICE']), target.to(config['DEVICE'])

        score = model(data)
        loss = criterion(score, target)
        loss_data = loss.item()
        val_loss.append(loss_data)
        # accv = (torch.max(score.cpu(), 1)[1] == target.cpu()).float().mean()
        output = torch.argmax(F.softmax(score, dim=1), dim=1)
        correct = torch.eq(output, target).int()
        accuracy = float(correct.sum()) / float(correct.numel())
        val_acc += accuracy
    acc_avg = val_acc / len(val_loader)


    if acc_avg > config['best_acc'] :
        print("saving model........")
        config['cnt'] = 0
        config['best_acc'] = acc_avg
        torch.save(model.state_dict(), './%s.pth' % config['timestr'])
    elif config['best_acc'] > acc_avg:
        config['cnt'] += 1

    print("valid_acc: ", acc_avg, "best_acc: ", config['best_acc'])
    print("valid loss: ",np.mean(val_loss))


def main():
    print("cuda:",torch.cuda.is_available())
    print("time: ",config['timestr'])
    train_ld, valid_ld = get_dataloaders()
    model = FCN32(num_classes=7).to(config['DEVICE'])
    print(model)
    # model = UNet().to(config['DEVICE'])
    if config['pretrained']:
        print("load state dict")
        model.load_state_dict(torch.load(config['pretrained']))

    for itr in range(config['epochs']):
        trainer(model, train_ld,valid_ld)

        if itr in[1,config['epochs']/2,config['epochs']]:
            # torch.save(model.state_dict(), './%s.pth' % config['timestr'])
            torch.save(model.state_dict(), f"{config['timestr']}_ep{itr}.pth")

        if config['cnt'] > 10:
            print("early stop!!!")
            break


if __name__ == '__main__':
    main()
