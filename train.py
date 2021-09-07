import torch
import numpy as np
from sklearn import metrics
import time


def train(train_iter, dev_iter, model, optimizer, dev_steps, num_epochs):
    # start_time = time.time()
    model.train()
    loss_func = torch.nn.CrossEntropyLoss()
    flag = False
    dev_best_loss = float('inf')
    total_batch = 0
    for epoch in range(num_epochs):
        for step, (inputs, labels) in enumerate(train_iter):
            outputs = model(inputs)
            model.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % dev_steps == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(model, dev_iter)
                # if dev_loss < dev_best_loss:
                #     dev_best_loss = dev_loss
                #     torch.save(model.state_dict(), config.save_path)
                #     improve = '*'
                #     last_improve = total_batch
                # else:
                #     improve = ''
                # time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, 0, 0))
                model.train()
            total_batch += 1
            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过1000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
        if flag:
            break


def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_func = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in data_iter:
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    # if test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
