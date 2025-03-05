import os
import pickle
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch_num", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--optimizer_name", type=str, default="DualAdam")

    args = parser.parse_args()

    utils.seed_torch(42)

    accuracies = []
    losses = []

    epoch_num = args.epoch_num
    lr = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = args.dataset_name

    trainloader, testloader = utils.load_data(dataset_name)

    model_name = args.model_name
    model = utils.select_model(model_name, device)

    if device == torch.device("cuda") and torch.cuda.device_count() > 1:
        print("Parallel Mode")
        model = torch.nn.DataParallel(model)

    optimizer_name = args.optimizer_name
    optimizer = utils.select_optimizer(optimizer_name, model, lr)

    # 实例化损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    print(f"Dataset:{dataset_name} Model: {model_name} Optimizer:{optimizer_name}")

    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr/1000, verbose=True)

    training_times = []

    for epoch in range(epoch_num):
        start_time = time.time()
        loss = utils.train(model, trainloader, optimizer, criterion, device=device)
        training_time = time.time() - start_time
        training_times.append(training_time)
        accuracy = utils.test(model, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy} LR:{ scheduler.get_last_lr()[0]} Training Time:{training_time}")

    # save the test accuracy over epoch
    with open('test_accuracy_over_epoch.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # save the training loss over epoch
    with open('training_loss_over_epoch.pkl', 'wb') as file:
        pickle.dump(losses, file)


