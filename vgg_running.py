from PMCModel import PMDatasets, VGG, Evaluator, Learner, Runner, BaseClassifier, NormalHead, PMLogger
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import cross_entropy
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
num_classes = 3

transforms = Compose([ToTensor(), Resize((224, 224))])

training_dataset = PMDatasets(data_path='data/imagenet', data_type='train', transforms=transforms)
testing_dataset = PMDatasets(data_path='data/imagenet', data_type='test', transforms=transforms)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

## experientment part

model_list = ['vgg19']

for model_name in model_list:
    head = NormalHead(3)
    backbone = VGG(model_name=model_name)
    classifier = BaseClassifier(model_name, backbone, head, cross_entropy)
    classifier = classifier.to(device)

    logger = PMLogger(path='work_dir', model_name=model_name)

    optimizer = optim.Adam(classifier.parameters())

    learner = Learner(training_dataloader, classifier, optimizer, logger, device)
    evaluator = Evaluator(testing_dataloader, classifier, logger, device)

    runner = Runner(evaluator, learner, logger, max_epoch=30)

    runner.run()
