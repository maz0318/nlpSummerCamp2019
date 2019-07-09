import torchvision

res = torchvision.models.resnet152()
print(list(res.children())[:-2])