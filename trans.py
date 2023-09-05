import os
import torch
import torchvision
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset
from util import transform_convert

datasets_root_dir = 'E:/workspace/Dataset'      # 想要存储数据集的目录
datasets_saved_dir = datasets_root_dir + '/cifar-10_jpg'
if not os.path.exists(datasets_saved_dir + '/train'):
    os.makedirs(datasets_saved_dir + '/train')
if not os.path.exists(datasets_saved_dir + '/test'):
    os.makedirs(datasets_saved_dir + '/test')
if not os.path.exists(datasets_saved_dir + '/train_test_pth'):
    os.makedirs(datasets_saved_dir + '/train_test_pth')


dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    ToTensor()
])

transform_test = Compose([
    ToTensor()
])

trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=1,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=1,
    shuffle=False)

train_samples = torch.zeros((50000, 3, 32, 32))
test_samples = torch.zeros((10000, 3, 32, 32))

train_labels = torch.zeros((50000, 1))
test_labels = torch.zeros((10000, 1))

for batch_id, batch in enumerate(train_loader):
    batch_img = batch[0]
    # print(batch_img)
    # print(batch_img.shape)
    batch_label = batch[1]
    train_samples[batch_id, :, :, :] = batch_img
    train_labels[batch_id] = batch_label
    if (batch_id + 1) % 5000 == 0:
        print((batch_id + 1) / 5000)
print("train ok")

for batch_id, batch in enumerate(test_loader):
    batch_img = batch[0]
    # print(batch_img)
    # print(batch_img.shape)
    batch_label = batch[1]
    test_samples[batch_id, :, :, :] = batch_img
    test_labels[batch_id] = batch_label
    if (batch_id + 1) % 1000 == 0:
        print((batch_id + 1) / 1000)
print("test ok")

torch.save(train_samples, datasets_saved_dir + '/train_test_pth/train_samples.pth')     # 使用torch.load()读取
torch.save(train_labels, datasets_saved_dir + '/train_test_pth/train_labels.pth')
torch.save(test_samples, datasets_saved_dir + '/train_test_pth/test_samples.pth')
torch.save(test_labels, datasets_saved_dir + '/train_test_pth/test_labels.pth')


with open(datasets_saved_dir + "/train_path.txt", "w") as f:
    f.close()
for i in range(50000):
    # print(train_samples[i])
    # print(train_samples[i].shape)
    img = transform_convert(train_samples[i].cpu(), transform_train)
    # img = Image.fromarray(img.permute(1, 2, 0).numpy())
    saved_path = datasets_saved_dir + '/train/train_' + str(i) + '.jpg'
    with open(datasets_saved_dir + "/train_path.txt", "a") as f:
        f.write(str(int(train_labels[i])) + '    ' + saved_path + '\n')  # 自带文件关闭功能，不需要再写f.close()
    img.save(saved_path)
    if (i + 1) % 5000 == 0:
        print((i + 1) / 5000)
print("save train ok")


with open(datasets_saved_dir + "/test_path.txt", "w") as f:
    f.close()
for i in range(10000):
    img = transform_convert(test_samples[i].cpu(), transform_test)
    # img = Image.fromarray(img.permute(1, 2, 0).numpy())
    saved_path = datasets_saved_dir + '/test/test_' + str(i) + '.jpg'
    with open(datasets_saved_dir + "/test_path.txt", "a") as f:
        f.write(str(int(test_labels[i])) + '    ' + saved_path + '\n')  # 自带文件关闭功能，不需要再写f.close()
    img.save(saved_path)
    if (i + 1) % 1000 == 0:
        print((i + 1) / 1000)
print("save test ok")
