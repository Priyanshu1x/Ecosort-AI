import numpy as np
import random
import torch

#set seed value
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(42)

from ecosort import dataset
from ecosort import evaluate
from ecosort import models
from ecosort import train

img_path,label=dataset.labeling()
train_dataloader,valid_dataloader,test_dataloader=dataset.create_dataloaders(img_path=img_path,label=label,batch_size=32)
model1=models.BaselineCNN()
model2=models.build_resnet18()
model3=models.build_EfficientNet()

final_model1,fulldataset1=train.train_model(model1,train_dataloader,valid_dataloader,0.001,40,checkpoint_path = "D:/Ecosort AI/checkpoints/baseline.pth")
final_model2,fulldataset2=train.train_model(model2,train_dataloader,valid_dataloader,0.001,15,checkpoint_path = "D:/Ecosort AI/checkpoints/resnet18.pth")
final_model3,fulldataset3=train.train_model(model3,train_dataloader,valid_dataloader,0.001,20,checkpoint_path = "D:/Ecosort AI/checkpoints/efficientnet.pth")

class_name=["Dry", "Wet", "Recyclable", "ewaste"]
accuracy1, y_true1, y_confd1, y_alter1,y_alter_class1, y_pred1, cm1=evaluate.evaluate_model(final_model1,test_dataloader,class_names=class_name)
accuracy2, y_true2, y_confd2, y_alter2,y_alter_class2, y_pred2, cm2=evaluate.evaluate_model(final_model2,test_dataloader,class_names=class_name)
accuracy3, y_true3, y_confd3, y_alter3,y_alter_class3, y_pred3, cm3=evaluate.evaluate_model(final_model3,test_dataloader,class_names=class_name)