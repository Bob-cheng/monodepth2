import os, socket
import torch

from torch.utils.data import Dataset, DataLoader
from dataLoader import KittiLoader
from cam_extraction import CarRecognition, save_model
from depth_model import import_depth_model
from tensorboardX import SummaryWriter
from datetime import datetime


import config
import argparse

image_size = (1024, 320) # w, h

parser = argparse.ArgumentParser(description='CAM vehicle classification retraining')
parser.add_argument('--learning-rate', '-lr', dest='lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help="How many epochs do you want to train")
parser.add_argument('-bs', default=16, type=int, help="batch size")

args = parser.parse_args()
print(args)

kitti_loader_train = KittiLoader(mode='train', size=image_size)
kitti_loader_eval = KittiLoader(mode='val', size=image_size)
train_loader = DataLoader(kitti_loader_train, batch_size=args.bs, shuffle=True, num_workers=5, pin_memory=True)
test_loader = DataLoader(kitti_loader_eval, batch_size=args.bs, shuffle=False, num_workers=5, pin_memory=True)

depth_model = import_depth_model(image_size).to(config.device0).eval()
recog_model = CarRecognition(depth_model, 2).to(config.device0)
for param in recog_model.depth_encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(recog_model.fc.parameters(), lr=args.lr)

log_name = datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname()
log_dir = os.path.join(os.path.abspath(os.getcwd()), 'CAM_logs', log_name)
os.makedirs(log_dir)
logger = SummaryWriter(log_dir)

logger.add_text('args/CLI_params', str(args), 0)

loss_func = torch.nn.CrossEntropyLoss()
global_steps = 0
log_interval = 300
for epoch in range(args.epochs):
    print('Epoch', epoch, 'train in progress...')
    
    # training
    recog_model.fc.train()
    for color, target in train_loader:
        color = color.to(config.device0)
        target = target.to(config.device0)
        predict = recog_model(color)
        # print(target.size())
        # print(predict.size())
        loss = loss_func(predict, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_steps += 1
            
    logger.add_scalar('train/loss', loss.item(), global_steps)

    # testing
    recog_model.fc.eval()
    eval_loss = 0
    eval_count = 0
    correct_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        for color, target in test_loader:
            color, target = color.to(config.device0), target.to(config.device0)
            predict = recog_model(color)
            loss = loss_func(predict, target)
            eval_loss += loss
            eval_count += 1
            pred_label = torch.argmax(predict, dim=1)
            correct_cnt += torch.sum(pred_label == target)
            total_cnt += target.size()[0]

    logger.add_scalar('test/loss',      eval_loss/eval_count,  global_steps)
    logger.add_scalar('test/accuracy',  correct_cnt/total_cnt, global_steps)

save_model(log_name, recog_model, optimizer)
logger.close()

