import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from dataset import CustomDataset
from utils import Resize, RandomCrop, RandomHorizontalFlip, ToTensor, show_keypoints
import numpy as np
import json


def collate_function(x):
    return list(zip(*x))


def train():
    epochs = 100

    # Get cpu or gpu device for training.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # Init model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, num_keypoints=14)
    model = model.to(device)

    # model.load_state_dict(torch.load('model.pth', map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Train Data
    prop = 1.1
    train_tf = transforms.Compose([Resize((int(160*prop), int(120*prop))),
                                   RandomCrop((160, 120)),
                                   RandomHorizontalFlip(),
                                   ToTensor()])

    train_dataset = CustomDataset('C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\train', ['.png'], train_tf, image_type='IR', cover_types=['cover1_gen', 'cover2_gen'])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_function)

    # Validation Data
    val_tf = transforms.Compose([ToTensor()])

    val_dataset = CustomDataset('C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\valid', ['.png'], val_tf, image_type='IR', cover_types=['cover1', 'cover2'])

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_function)

    best_loss = np.inf

    for e in range(epochs):
        model.train()

        accom_loss = 0
        loss_interval = 1

        for batch_idx, data in enumerate(train_loader):
            images, pose = data[0], data[1]
            images = list(image.to(device) for image in images)
            pose = [{k: v.to(device) for k, v in t.items()} for t in pose]

            # Compute prediction error
            pred = model(images, pose)
            loss = sum(losses for losses in pred.values())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accom_loss += loss.item()

            if (batch_idx + 1) % loss_interval == 0:
                print(f'epoch: {e}, batch: {batch_idx}, loss: {accom_loss / loss_interval}')
                accom_loss = 0

        model.eval()

        predicted_keypoints = []
        gt_keypoints = []

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                images, pose = data[0], data[1]
                images = list(image.to(device) for image in images)
                pose = [{k: v.to(device) for k, v in t.items()} for t in pose]

                # Compute prediction error
                pred = model(images)

                for i in range(len(pred)):
                    pred_scores = pred[i]['scores']
                    pred_keypoints = pred[i]['keypoints']

                    max_score_idx = torch.max(pred_scores, dim=0)[1]

                    pred_keypoints = pred_keypoints[max_score_idx]

                    predicted_keypoints.append(pred_keypoints[:, 0:2].cpu())
                    gt_keypoints.append(pose[i]['keypoints'][0][:, 0:2].cpu())

        predicted_keypoints = torch.stack(predicted_keypoints, dim=0)
        gt_keypoints = torch.stack(gt_keypoints, dim=0)

        mse = torch.nn.MSELoss()
        val_loss = mse(predicted_keypoints, gt_keypoints)

        print(f'best val loss: {best_loss}, current val loss: {val_loss}')

        if val_loss.item() <= best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'model.pth')
            print('saved model!')


def test():
    # Get cpu or gpu device for training.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # Init model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, num_keypoints=14)
    model = model.to(device)

    model.load_state_dict(torch.load('model.pth', map_location=device))

    # Data
    tf = transforms.Compose([ToTensor()])

    dataset = CustomDataset('C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\valid', ['.png'], tf, image_type='IR', cover_types=['cover1', 'cover2'])

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_function)

    # Loop
    model.eval()

    with torch.no_grad():

        for batch_idx, data in enumerate(loader):
            images, pose = data[0], data[1]
            images = list(image.to(device) for image in images)
            pose = [{k: v.to(device) for k, v in t.items()} for t in pose]

            # Compute prediction error
            pred = model(images)

            for i in range(len(pred)):
                pred_scores = pred[i]['scores']
                pred_keypoints = pred[i]['keypoints']

                max_score_idx = torch.max(pred_scores, dim=0)[1]

                pred_keypoints = pred_keypoints[max_score_idx]

                show_keypoints(images[i].cpu(), pred_keypoints.cpu())


def gen_submission():
    # Get cpu or gpu device for training.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # Init model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, num_keypoints=14)
    model = model.to(device)

    model.load_state_dict(torch.load('model.pth', map_location=device))

    # Data
    tf = transforms.Compose([transforms.ToTensor()])

    dataset = CustomDataset('C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\test1', ['.png'], tf, image_type='IR', cover_types=['cover1', 'cover2'])

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Loop
    model.eval()

    results = []

    with torch.no_grad():

        for batch_idx, images in enumerate(loader):
            images = list(image.to(device) for image in images)

            # Compute prediction error
            pred = model(images)

            for i in range(len(pred)):
                pred_scores = pred[i]['scores']
                pred_keypoints = pred[i]['keypoints']

                max_score_idx = torch.max(pred_scores, dim=0)[1]

                pred_keypoints = pred_keypoints[max_score_idx]

                results.append(pred_keypoints[:, 0:2].cpu().numpy())

    results = np.array(results)
    results = results.tolist()

    with open('preds.json', 'w') as json_file:
        json.dump(results, json_file)


if __name__ == '__main__':
    train()


