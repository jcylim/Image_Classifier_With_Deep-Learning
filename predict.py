# Necessary packages for execution of this script
import argparse 

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import json
# import necessary packages for building and trainig network
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict

class LoadCheckpoint:
    def __init__(self, imgPath, filepath):
        self.imgPath = imgPath
        self.filepath = filepath
    
    def load_model(self):
        checkpoint = torch.load(self.filepath)
        model_name = checkpoint['nn']
        model = getattr(models, model_name)(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.epochs = checkpoint['epochs']
        learning_rate = checkpoint['learning_rate']
        model.optimizer = checkpoint['optimizer']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
        return model
    
    def process_image(self):
        original_img = Image.open(self.imgPath)
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
        img = transform(original_img)
        return img
    
class Prediction:
    def __init__(self, model, img, topk, cat_to_name, gpu):
        self.model = model
        self.img = img
        self.topk = 1 if not topk else topk
        self.cat_to_name = 'cat_to_name.json' if not cat_to_name else cat_to_name
        with open(self.cat_to_name, 'r') as f:
            self.cat_to_name = json.load(f)
        self.gpu = 'cuda' if gpu == True else 'cpu'
        self.device = torch.device(self.gpu)
        
    def predict(self):
        # Implement the code to predict the class from an image file
        self.model.to(self.device)
        self.model.eval()

        self.img = self.img.unsqueeze_(0)
        self.img = self.img.float()

        classMap = self.model.class_to_idx

        with torch.no_grad():
            logps = self.model(self.img.cuda())
            ps = torch.exp(logps)
            top_p, top_class_idx = ps.topk(self.topk, dim=1)
            top_class_idx = [k for k, v in classMap.items() if v in np.array(top_class_idx)] 
            
        top_classes = [self.cat_to_name[x] for x in top_class_idx]
        
        return np.array(top_p)[0], top_classes
        
if __name__ == '__main__':
    # Args for python script
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help="Path to image for prediction.", type=str)
    parser.add_argument('checkpoint', help="Path to chekpoint file from pre-trained model for prediction.", type=str)
    parser.add_argument('--top_k', help="Number of most probable classes predicted.", type=int)
    parser.add_argument('--category_names', help="Map used to map category names to predicted classes.", type=str)
    parser.add_argument('--gpu', help="Use GPU to predict model.", action='store_true')
    args = parser.parse_args()
    
    # Load and process image for prediction
    loadCP = LoadCheckpoint(args.img_path, args.checkpoint)
    model = loadCP.load_model()
    img = loadCP.process_image()
    
    # Prediction/Inference
    prediction = Prediction(model, img, args.top_k, args.category_names, args.gpu)
    ps, classes = prediction.predict()
    print("Flower name(s): {}".format(classes)) # top classes 
    print("Class probability: {}".format(ps)) # top probabilities
    
    
    