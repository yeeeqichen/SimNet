import torch


def train(dataloader, model, loss_func, lr, num_epochs, device, log_steps):
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    for epoch in range(num_epochs):
        step = 0
        hit = 0
        total = 0.
        for ids_1, masks_1, ids_2, masks_2, labels in dataloader.get_batch_data():
            outputs_1, outputs_2 = model(ids_1, masks_1, ids_2, masks_2)
            loss, predict_results = loss_func(outputs_1, outputs_2, labels)
            predict_labels = torch.argmax(predict_results, dim=1)
            labels = torch.tensor(labels).to(device)
            hit += (predict_labels == labels).sum().float()
            total += labels.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % log_steps == 0:
                print('EPOCH: {}, STEPS: {}, LOSS: {}, ACC: {}'.format(epoch, step, loss, hit / total))
                hit = 0
                total = 0.




