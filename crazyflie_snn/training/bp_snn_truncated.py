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
    # target = torch.zeros_like(target)
    # for i in range(1, len(target)):
        # target[i, :, 0] = target[i-1, :, 0] + ((data[:, :, 0][i] - data[:, :, 7][i] * (3/5)) * 6 - data[:, :, 3][i]) * 0.5 * 0.0008
        # # target[i, :, 1] = target[i-1, :, 1] + ((data[:, :, 1][i] - data[:, :, 6][i] * (3/5)) * 6 + data[:, :, 4][i]) * 0.5 * 0.0008
        # target[i, :, 0] = target[i-1, :, 0] + target[:, :, 0][i] * 0.0008
        # target[i, :, 1] = target[i-1, :, 1] + data[:, :, 1][i] * 0.0008
        # target[i, :, 0] = data[:, :, 0][i] - data[:, :, 2][i]
        # target[i, :, 1] = data[:, :, 1][i] - data[:, :, 3][i]
    model.reset(device=device)
    output = model.forward_sequence(data)
    loss = loss_function(target, output)
    # diff_loss = loss_function(torch.diff(output, dim=0), torch.diff(target, dim=0)) * 20000
    # mse = torch.functional.F.mse_loss(output, target)
    # print(diff_loss, mse)
    # loss = diff_loss + mse
    return loss, [data, output, target]

def train(model, device, train_loader, optimizer, t, max_batches, loss_function, plot_results=False):
    model.train()
    losses = []
    torch.seed()
    for i in range(min(max_batches, len(train_loader))):
        model.reset(device=device)

        # loss, (data, output, target) = eval_model(model, train_loader, loss_function, device)

        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.permute(1, 0, 2)
        target = target.permute(1, 0, 2)

        step_trunc = 0
        K_count = 0
        K_trunc = 50
        output_trunc = []
        output_total = []
        for step in range(len(data)):
            output_step = model(data[step])
            output_trunc.append(output_step)
            output_total.append(output_step.detach().clone())
            step_trunc += 1

            if step_trunc == K_trunc:
                output = torch.stack(output_trunc)
                loss = loss_function(target[int(K_count * K_trunc):int((K_count+1) * K_trunc)], output)
                losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # detach hidden states
                model.detach_hidden()

                K_count += 1
                step_trunc = 0
                output_trunc = []


                
        # model.mask_grads()

        # Clip gradients and perform optimization step
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        # print all parameters
        # for name, param in model.named_parameters():
        #     print(name, param.grad.abs().max())
        

        output = torch.stack(output_total)
        if plot_results:
            plt.clf()
            n_plots = target.size()[2]
            for i in range(n_plots):
                plt.subplot(n_plots, 1, i + 1)
                plt.plot(output[:, 0, i].cpu().detach().numpy(), label="snn")
                plt.plot(target[:, 0, i].cpu().detach().numpy(), label="target")
                plt.legend()
            plt.ylim([-1,1])
            # calc_integ = np.zeros(len(data[:, 0, 0]))
            # for i in range(1, len(data[:, 0, 0])):
            #     # calc_integ[i] = calc_integ[i-1] + ((data[:, 0, 0][i] - data[:, 0, 7][i] * (3/5)) * 6 - data[:, 0, 3][i]) * 0.5 * 0.0008
            #     calc_integ[i] = calc_integ[i-1] + data[:, 0, 0][i] * 0.002
                # calc_integ[i] = 0.99 * calc_integ[i-1] + 0.01 * (data[:, 0, 4][i] - (data[:, 0, 0][i] - data[:, 0, 2][i]) * 0.005)
            # plt.subplot(2, 1, 1)
            # plt.plot(calc_integ, label="roll integ")
            # plt.ylim([-0.5, 0.5])

            # calc_integ = np.zeros(len(data[:, 0, 0]))
            # for i in range(1, len(data[:, 0, 0])):
            #     # calc_integ[i] = calc_integ[i-1] + ((data[:, 0, 1][i] - data[:, 0, 6][i] * (3/5)) * 6 + data[:, 0, 4][i]) * 0.5* 0.0008
            #     calc_integ[i] = calc_integ[i-1] + data[:, 0, 1][i] * 0.002
            #     # calc_integ[i] = 0.99 * calc_integ[i-1] + 0.01 * (data[:, 0, 5][i] - (data[:, 0, 1][i] - data[:, 0, 3][i]) * 0.005)
            # plt.subplot(2, 1, 2)
            # plt.plot(calc_integ, label="pitch integ")
            # plt.ylim([-0.5, 0.5])
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
    torch.autograd.set_detect_anomaly(True)
    try:
        with tqdm(total=EPOCHS) as t_outer:
            for epoch in range(EPOCHS):
                epoch_t = time()
                with tqdm(total=min(max_batches, len(train_loader))) as t:
                # t=0
                    mean_loss = train(model, device, train_loader, optimizer, t, max_batches, loss_function, plot_results=config["plot_results"])
                    test_loss = test(model, device, val_loader, loss_function)
                    mean_losses.append(float(mean_loss))
                    test_losses.append(float(test_loss))
                    time_hist.append(time() - epoch_t)
                    
                    if test_loss < best_loss:
                        best_model = deepcopy(model.state_dict())
                        if out_dir is not None:
                            model_save_file = f'{out_dir}/model.pt'
                            with open(model_save_file, 'wb') as f:
                                torch.save(best_model, f)
                        best_loss = mean_loss
                    t.set_postfix(ordered_dict={"mean_loss":"{:05.4f}".format(mean_loss), "val_loss":"{:05.4f}".format(test_loss)})
                t_outer.update()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    if config["plot_results"]:
        plt.show()
    print(f"Best test loss: {float(np.min(test_losses))}")
    best_fitness = min(mean_losses)
    avg_time = float(np.mean(time_hist))
    return best_model, best_fitness, epoch, avg_time, mean_losses, test_losses