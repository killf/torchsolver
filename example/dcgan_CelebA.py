import torchsolver as ts
from torchvision.transforms import *

transform = Compose([CenterCrop(128), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = ts.datasets.ImageFiles("/data2/dataset/CelebA/Img/img_align_celeba/img_align_celeba", transform=transform)
ts.nets.DCGANNet(task_name="dcgan_CelebA", image_size=(3, 128, 128), epochs=50, batch_size=128).fit(train_data=train_data)
