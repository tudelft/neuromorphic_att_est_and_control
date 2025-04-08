import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
from tqdm import tqdm
from copy import deepcopy
import wandb

def eval_model(model, dataloader, loss_function, device):
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)
    data = data.permute(1, 0, 2)
    target = target.permute(1, 0, 2)
    model.reset(device=device)
    output = model.forward_sequence(data)
    loss = loss_function(target, output)
    return loss, [data, output, target]

def train(model, device, train_loader, optimizer, t, max_batches, loss_function, plot_results=False):
    model.train()
    losses = []
    torch.seed()
    for i in range(min(max_batches, len(train_loader))):
        optimizer.zero_grad()
        loss, (data, output, target) = eval_model(model, train_loader, loss_function, device)
        # Throw error if it contains NaN
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        loss.backward()
        if model.do_mask_grads is not None:
            model.mask_grads()

        # Clip gradients and perform optimization step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        # print all parameters
        # for name, param in model.named_parameters():
        #     print(name, param.grad.abs().max())
        

        losses.append(loss.item())
        
        if plot_results:
            plt.clf()
            n_plots = target.size()[2] + 2
            for i in range(n_plots - 2):
                plt.subplot(n_plots, 1, i + 1)
                plt.plot(output[:, 0, i].cpu().detach().numpy(), label="snn")
                plt.plot(target[:, 0, i].cpu().detach().numpy(), label="target")
                plt.ylim([-1,1])
                plt.legend()
            
            # plt.subplot(n_plots, 3, n_plots * 3 - 5)
            # plt.hist(model.l1.state[2][0].flatten().cpu().detach().numpy(), bins=80, label="i")
            # plt.subplot(n_plots, 3, n_plots * 3 - 4)
            # plt.hist(model.l1.state[2][1].flatten().cpu().detach().numpy(), bins=80, label="v")
            # plt.subplot(n_plots, 3, n_plots * 3 - 3)
            # # plt.hist(model.l1.state[2][2].flatten().cpu().detach().numpy(), bins=80, label="s")
            # plt.plot(model.l1.state[2][2].sum(dim=0).cpu().detach().numpy(), '.', label="s")
            # plt.ylim([0, 100])

            # plt.subplot(n_plots, 2, n_plots * 2 - 1)
            # plt.hist(model.l1.state[0][0].flatten().cpu().detach().numpy(), bins=80, label="ff")
            # plt.subplot(n_plots, 2, n_plots * 2)
            # plt.hist(model.l1.state[1][0].flatten().cpu().detach().numpy(), bins=80, label="rec")

            plt.pause(0.02)

        # update the loading bar
        t.set_postfix(loss="{:05.4f}".format(loss.item()))
        wandb.log({"loss": loss.item()})
        t.update()

    mean_loss = np.mean(losses)
    t.set_postfix(mean_loss="{:05.4f}".format(mean_loss))
    return mean_loss


def test(model, device, test_loader, loss_function):
    model.eval()
    loss, _ = eval_model(model, test_loader, loss_function, device)
    wandb.log({"val_loss": float(loss)})
    return loss

def fit(model, train_loader, val_loader, optimizer, config, out_dir, loss_function):
    mean_losses = []
    test_losses = []
    time_hist = []
    best_model = None
    best_loss = np.inf
    max_batches = 10
    EPOCHS = config["train"]["gens"]
    device = config["device"]
    # torch.autograd.set_detect_anomaly(True)
    try:
        with tqdm(total=EPOCHS) as t_outer:
            for epoch in range(EPOCHS):
                epoch_t = time()
                with tqdm(total=min(max_batches, len(train_loader))) as t:
                # t=0
                    try:
                        mean_loss = train(model, device, train_loader, optimizer, t, max_batches, loss_function, plot_results=config["plot_results"])
                    except ValueError:
                        print("NaN Value in loss")
                        break
                    test_loss = test(model, device, val_loader, loss_function)
                    mean_losses.append(float(mean_loss))
                    test_losses.append(float(test_loss))
                    time_hist.append(time() - epoch_t)
                    
                    updated_model = False
                    if test_loss < best_loss:
                        updated_model = True
                        best_model = deepcopy(model.state_dict())
                        if out_dir is not None:
                            model_save_file = f'{out_dir}/model.pt'
                            with open(model_save_file, 'wb') as f:
                                torch.save(best_model, f)
                        best_loss = test_loss
                    t.set_postfix(ordered_dict={"mean_loss":"{:05.4f}".format(mean_loss), "val_loss":"{:05.4f}".format(test_loss), "updated model":updated_model})
                t_outer.update()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    if config["plot_results"]:
        plt.show()

    if len(test_losses) > 0:
        print(f"Best test loss: {float(np.min(test_losses))}")
    best_fitness = min(mean_losses)
    avg_time = float(np.mean(time_hist))
    return best_model, best_fitness, epoch, avg_time, mean_losses, test_losses