import sys, os
import torch
import torch.nn
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
from model import MyNet
from data import Bike
from torch.utils.data import  DataLoader
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
if __name__ == "__main__":
    # Specifiy data folder path and model type
    folder, model_type = sys.argv[1], sys.argv[2]
    print("cuda:　", torch.cuda.is_available())
    timestr = time.strftime("%H%M%S")
    print("time: ", timestr)

    if model_type == 'conv':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(size=(299, 299)),
            transforms.RandomAdjustSharpness(sharpness_factor=0),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ColorJitter(brightness=.5, contrast=.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # model = models.vgg19_bn(pretrained=True)
        # model.classifier[6].out_features = 50
        model = models.inception_v3(pretrained = True)
        model.fc.out_features = 50
        print(model)

    elif model_type == 'mynet':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(size=(96, 96)),
            transforms.RandomAdjustSharpness(sharpness_factor=0),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ColorJitter(brightness=.5, contrast=.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        model = MyNet()
    print(model)
    # Get data loaders of training set and validation set
    train_set, val_set = Bike(root_dir=os.path.join(folder, "train_50"), transform=transform), Bike(
        root_dir=os.path.join(folder, "val_50"), transform=transform)
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=8, shuffle=True)

    # Set the type of gradient optimizer and the model it update
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1, weight_decay=1e-3)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # Choose loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load('021801.pth'))

    ep = 50
    train_loss = []
    valid_loss = []
    valid_acc = []
    train_acc = []
    best_acc = 0
    cnt = 0
    for epoch in range(ep):
        print('Epoch:', epoch)

        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0

        # Load batch data from dataloader
        for batch, (x, label) in enumerate(tqdm(train_loader),1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            x, label = x.to(device), label.to(device)
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 250 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch
                print('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
                train_acc.append(acc)
                train_loss.append(ave_loss)

        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        loss_val = 0
        total_correct = 0
        total_cnt = 0
        accuracy = 0
        for batch, (x, label) in enumerate(val_loader, 1):
            x, label = x.to(device), label.to(device)
            out = model(x)
            loss = criterion(out, label)
            loss_val += loss.item()
            _, pred_label = torch.max(out, 1)
            total_correct += (pred_label == label).sum().item()
            total_cnt += x.size(0)
            if batch % 150 == 0 or batch == len(val_loader):
                loss_val = loss_val / batch
                accuracy = total_correct / total_cnt
                valid_acc.append(accuracy)
                valid_loss.append(loss_val)
        print("current loss: ", loss_val, "   valid_acc: ", accuracy)
        print("best_acc ", best_acc)
        if accuracy > best_acc:
            cnt = 0
            best_acc = accuracy
            torch.save(model.state_dict(), './%s.pth' % timestr)
        elif best_acc > accuracy:
            cnt += 1
        if cnt > 10:
            print("early stop!!!")
            break
        if epoch in[0,15]:
            torch.save(model.state_dict(), './%s_ep%s.pth' % (timestr,epoch))


            # Calculate the training loss and accuracy of each iteration

            # Show the training information
        # scheduler.step()
        model.train()

    # Save trained model
    #
    # # Plot Learning Curve
    #
    # fig, ax = plt.subplots(2, 1)
    # ax[0].set_title("accuracy")
    # # ax[0].legend(loc="upper right")
    # ax[0].plot(train_acc, color='green')
    # ax[0].plot(valid_acc, color='red')
    #
    # ax[1].set_title("loss")
    # # ax[1].legend(loc="upper right")
    # ax[1].plot(train_loss, color='green')
    # ax[1].plot(valid_loss, color='red')
    #
    # # ax[1,0].set_title("valid_accuracy")
    # # ax[1,0].plot(valid_acc)
    # # ax[1,1].set_title("valid_loss")
    # # ax[1,1].plot(valid_loss)
    # plt.show()
