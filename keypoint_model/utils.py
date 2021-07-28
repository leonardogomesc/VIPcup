import torch
import torchvision.transforms as transforms
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Resize:
    def __init__(self, output_size):
        self.final_height, self.final_width = output_size
        self.tf = transforms.Resize(output_size)

    def __call__(self, sample):
        image = sample[0]
        target = sample[1]

        keypoints = target['keypoints']
        boxes = target['boxes']
        labels = target['labels']

        width, height = image.size

        wp = self.final_width / width
        hp = self.final_height / height

        image = self.tf(image)

        keypoints[:, :, 0] *= wp
        keypoints[:, :, 1] *= hp

        boxes[:, 0] *= wp
        boxes[:, 2] *= wp
        boxes[:, 1] *= hp
        boxes[:, 3] *= hp

        target = {'keypoints': keypoints, 'boxes': boxes, 'labels': labels}

        return image, target


class RandomCrop:
    def __init__(self, output_size):
        self.final_height, self.final_width = output_size

    def __call__(self, sample):
        image = sample[0]
        target = sample[1]

        keypoints = target['keypoints']
        boxes = target['boxes']
        labels = target['labels']

        width, height = image.size

        dw = width - self.final_width
        dh = height - self.final_height

        bb = [min(boxes[:, 0]), min(boxes[:, 1]), max(boxes[:, 2]), max(boxes[:, 3])]

        if bb[2] - bb[0] > self.final_width or bb[3] - bb[1] > self.final_height:
            left = random.randint(0, dw)
            top = random.randint(0, dh)
        else:
            min_x = max(0, math.ceil(bb[2] - self.final_width + 1))
            max_x = min(dw, math.floor(bb[0]))

            min_y = max(0, math.ceil(bb[3] - self.final_height + 1))
            max_y = min(dh, math.floor(bb[1]))

            left = random.randint(min_x, max_x)
            top = random.randint(min_y, max_y)

        image = image.crop((left, top, left + self.final_width, top + self.final_height))

        keypoints[:, :, 0] -= left
        keypoints[:, :, 1] -= top

        boxes[:, 0] -= left
        boxes[:, 2] -= left
        boxes[:, 1] -= top
        boxes[:, 3] -= top

        # change visibility if keypoint out of frame
        vis = (keypoints[:, :, 0] >= 0) & (keypoints[:, :, 0] < self.final_width) & (keypoints[:, :, 1] >= 0) & (keypoints[:, :, 1] < self.final_height) & (keypoints[:, :, 2] != 0)
        keypoints[:, :, 2] = torch.where(vis, 1, 0)

        # boxes must be within frame
        boxes[:, 0] = torch.where(boxes[:, 0] < 0, torch.tensor(0, dtype=boxes.dtype), boxes[:, 0])
        boxes[:, 2] = torch.where(boxes[:, 2] >= self.final_width, torch.tensor(self.final_width - 1, dtype=boxes.dtype), boxes[:, 2])
        boxes[:, 1] = torch.where(boxes[:, 1] < 0, torch.tensor(0, dtype=boxes.dtype), boxes[:, 1])
        boxes[:, 3] = torch.where(boxes[:, 3] >= self.final_height, torch.tensor(self.final_height - 1, dtype=boxes.dtype), boxes[:, 3])

        target = {'keypoints': keypoints, 'boxes': boxes, 'labels': labels}

        return image, target


class RandomHorizontalFlip:

    def __call__(self, sample):
        image = sample[0]
        target = sample[1]

        keypoints = target['keypoints']
        boxes = target['boxes']
        labels = target['labels']

        if random.randint(0, 1) == 1:
            width, height = image.size
            image = transforms.functional.hflip(image)

            keypoints[:, :, 0] = width - keypoints[:, :, 0] - 1

            keypoints_temp = torch.zeros_like(keypoints)

            keypoints_temp[:, 0] = keypoints[:, 5]  # Right ankle - 0
            keypoints_temp[:, 1] = keypoints[:, 4]  # Right knee - 1
            keypoints_temp[:, 2] = keypoints[:, 3]  # Right hip - 2
            keypoints_temp[:, 3] = keypoints[:, 2]  # Left hip - 3
            keypoints_temp[:, 4] = keypoints[:, 1]  # Left knee - 4
            keypoints_temp[:, 5] = keypoints[:, 0]  # Left ankle - 5
            keypoints_temp[:, 6] = keypoints[:, 11]  # Right wrist - 6
            keypoints_temp[:, 7] = keypoints[:, 10]  # Right elbow - 7
            keypoints_temp[:, 8] = keypoints[:, 9]  # Right shoulder - 8
            keypoints_temp[:, 9] = keypoints[:, 8]  # Left shoulder - 9
            keypoints_temp[:, 10] = keypoints[:, 7]  # Left elbow - 10
            keypoints_temp[:, 11] = keypoints[:, 6]  # Left wrist - 11
            keypoints_temp[:, 12] = keypoints[:, 12]  # Thorax - 12
            keypoints_temp[:, 13] = keypoints[:, 13]  # Head top - 13

            keypoints = keypoints_temp

            boxes[:, 0] = width - boxes[:, 0] - 1
            boxes[:, 2] = width - boxes[:, 2] - 1

            boxes_temp = torch.zeros_like(boxes)

            boxes_temp[:, 0] = boxes[:, 2]
            boxes_temp[:, 2] = boxes[:, 0]
            boxes_temp[:, 1] = boxes[:, 1]
            boxes_temp[:, 3] = boxes[:, 3]

            boxes = boxes_temp

        target = {'keypoints': keypoints, 'boxes': boxes, 'labels': labels}

        return image, target


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image = sample[0]
        target = sample[1]

        keypoints = target['keypoints']
        boxes = target['boxes']
        labels = target['labels']

        selected_degrees = random.uniform(self.degrees[0], self.degrees[1])
        width, height = image.size
        center = ((width - 1) / 2, (height - 1) / 2)
        fill = image.getpixel((10, 10))

        image = transforms.functional.rotate(image, selected_degrees, center=center, fill=fill)

        rads = (selected_degrees * math.pi) / 180
        rads = -rads

        center = torch.tensor(center)
        v = keypoints[:, :, 0:2] - center

        new_v = torch.zeros_like(v)

        new_v[:, :, 0] = math.cos(rads) * v[:, :, 0] - math.sin(rads) * v[:, :, 1]
        new_v[:, :, 1] = math.sin(rads) * v[:, :, 0] + math.cos(rads) * v[:, :, 1]

        keypoints[:, :, 0:2] = center + new_v

        boxes[:, 0] = torch.min(keypoints[:, :, 0], dim=1)[0]
        boxes[:, 1] = torch.min(keypoints[:, :, 1], dim=1)[0]
        boxes[:, 2] = torch.max(keypoints[:, :, 0], dim=1)[0]
        boxes[:, 3] = torch.max(keypoints[:, :, 1], dim=1)[0]

        # change visibility if keypoint out of frame
        vis = (keypoints[:, :, 0] >= 0) & (keypoints[:, :, 0] < width) & (keypoints[:, :, 1] >= 0) & (keypoints[:, :, 1] < height) & (keypoints[:, :, 2] != 0)
        keypoints[:, :, 2] = torch.where(vis, 1, 0)

        # boxes must be within frame
        boxes[:, 0] = torch.where(boxes[:, 0] < 0, torch.tensor(0, dtype=boxes.dtype), boxes[:, 0])
        boxes[:, 2] = torch.where(boxes[:, 2] >= width, torch.tensor(width - 1, dtype=boxes.dtype), boxes[:, 2])
        boxes[:, 1] = torch.where(boxes[:, 1] < 0, torch.tensor(0, dtype=boxes.dtype), boxes[:, 1])
        boxes[:, 3] = torch.where(boxes[:, 3] >= height, torch.tensor(height - 1, dtype=boxes.dtype), boxes[:, 3])

        target = {'keypoints': keypoints, 'boxes': boxes, 'labels': labels}

        return image, target


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.tf = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample):
        image = sample[0]
        target = sample[1]

        image = self.tf(image)

        return image, target


class ToTensor:
    def __init__(self):
        self.tf = transforms.ToTensor()

    def __call__(self, sample):
        image = sample[0]
        target = sample[1]

        image = self.tf(image)

        return image, target


def show_keypoints(image, coords):
    img = np.uint8(np.moveaxis(image.numpy(), 0, 2) * 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(len(coords)):
        cv2.circle(img, (int(coords[i, 0]), int(coords[i, 1])), 2, (0, 0, 255), thickness=-1)
        cv2.putText(img, f"{i}", (int(coords[i, 0] + 4), int(coords[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    plt.imshow(img)
    plt.show()

