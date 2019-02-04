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

class ProcessImage:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
    
    def load_data(self):
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
        test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
        validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        # Load the datasets with ImageFolder
        img_dataset = [datasets.ImageFolder(self.train_dir, transform=train_transforms),
                       datasets.ImageFolder(self.test_dir, transform=test_transforms),
                       datasets.ImageFolder(self.valid_dir, transform=validation_transforms)]

        # Using the image datasets and the trainforms, define the dataloaders
        dataloader = [torch.utils.data.DataLoader(img_dataset[0], batch_size=64, shuffle=True),
                      torch.utils.data.DataLoader(img_dataset[1], batch_size=64),
                      torch.utils.data.DataLoader(img_dataset[2], batch_size=64)]
        
        return img_dataset, dataloader
    
    def view_img(self, img_path):
        with Image.open(img_path) as image:
            plt.imshow(image)
                
class NeuralNetwork:
    def __init__(self, dataset, dataloader, arch, lr, hidden_units, epochs, gpu, save_dir):
        self.dataset = dataset
        self.dataloader = dataloader
        self.arch = 'densenet121' if not arch else arch
        self.lr = 0.001 if not lr else lr
        self.hidden_units = 512 if not hidden_units else hidden_units
        self.epochs = 5 if not epochs else epochs
        self.gpu = 'cuda' if gpu == True else 'cpu'
        self.save_dir = '' if not save_dir else save_dir
        self.device = torch.device(self.gpu)
        self.model, self.optimizer, self.criterion = self._setup_nn()
        
    def _setup_nn(self):
        model = getattr(models, self.arch)(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                    ('hidden1', nn.Linear(1024, self.hidden_units)),
                                    ('relu1', nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.3)),
                                    ('hidden2', nn.Linear(self.hidden_units, 256)),
                                    ('relu2', nn.ReLU()),
                                    ('dropout2', nn.Dropout(0.2)),
                                    ('output', nn.Linear(256, 102)),
                                    ('log_softmax', nn.LogSoftmax(dim=1))
                                    ]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.lr)
        
        return model, optimizer, criterion 
        
    def train(self):
        # Train the classifier using backprop and calculate for accuracy and loss
        self.model.to(self.device)
        step = 0
        print_every = 10
        
        for epoch in range(self.epochs):
            train_loss = 0
            for image, label in self.dataloader[0]:
                step += 1
                image, label = image.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                # Carry out steps for training neural network
                logits = self.model(image)
                loss = self.criterion(logits, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if step % print_every == 0:
                    img_accuracy = 0 
                    valid_loss = 0
                    
                    #Turn off dropouts 
                    self.model.eval()
                    
                    for image, label in self.dataloader[2]:
                        image, label = image.to(self.device), label.to(self.device)
                        # Turn off gradients to speed up process
                        with torch.no_grad():
                            logps = self.model(image)
                            loss = self.criterion(logps, label)
                            valid_loss += loss.item()
                            # Calculate accuracy and probability densities for each class
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == label.view(*top_class.shape)
                            accuracy = torch.mean(equals.type(torch.FloatTensor))
                            img_accuracy += accuracy.item()

                    print("@ Epoch {0}/{1}:".format(epoch+1, self.epochs), 
                          "Training Loss: {0}".format(train_loss/print_every),
                          "Validation Loss: {0}".format(valid_loss/len(self.dataloader[2])), 
                          "Accuracy Score: {0}".format(img_accuracy/len(self.dataloader[2]))) 

                    train_loss = 0 

                    # Turn on dropouts for training
                    self.model.train()
    
    def test(self):
        test_accuracy = 0
        self.model.to(self.device)
        
        self.model.eval()

        with torch.no_grad():
            for image, label in dataloader[1]:
                image, label = image.to(self.device), label.to(self.device)
                logps = self.model(image)               
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == label.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                test_accuracy += accuracy.item()

        print("Overall testing accuracy of this network is {:.3f}%".format((test_accuracy/len(dataloader[1]))*100))
    
    def save(self):
        # Save the checkpoint 
        self.model.class_to_idx = self.dataset[0].class_to_idx

        checkpoint = {'input_size': 1024,
                      'output_size': 102,
                      'epochs': self.epochs,
                      'nn': self.arch,
                      'classifier': self.model.classifier,
                      'learning_rate': self.lr,
                      'state_dict': self.model.state_dict(),
                      'class_to_idx': self.model.class_to_idx,
                      'optimizer': self.optimizer.state_dict()}
        
        filepath = self.save_dir + 'checkpoint.pth'
        torch.save(checkpoint, filepath)
        print("Chekpoints saved in path: {}".format(filepath))
        
if __name__ == '__main__':
    # Args for python script
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help="A dataset has to be provided to process data. The dataset directory is in the form: 'path/to/dataset'", type=str)
    parser.add_argument('--save_dir', help="Set directory to save checkpoints.", type=str)
    parser.add_argument('--arch', help="Choose a model architecture to train model on (default = densenet121).", type=str)
    parser.add_argument('--learning_rate', '--lr', help="Set learning rate for training model (default value = 0.001).", type=float)
    parser.add_argument('--hidden_units', help="Number of hidden nodes for fully-connected network/classifier.", type=int)
    parser.add_argument('--epochs', help="Number of epochs to train model.", type=int)
    parser.add_argument('--gpu', help="Use GPU to train model.", action='store_true')
    args = parser.parse_args()
    
    # ETL image data
    processData = ProcessImage(args.data_dir)
    img_dataset, dataloader = processData.load_data()
    
    # Train and save model
    model = NeuralNetwork(img_dataset, dataloader, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu, args.save_dir)
    model.train()
    model.test()
    model.save()
    
    