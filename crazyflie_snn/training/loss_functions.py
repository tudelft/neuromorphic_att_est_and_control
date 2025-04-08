import torch


def pearson(y_true, y_pred):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    pearson_loss = cos(
        y_pred - y_pred.mean(dim=0, keepdim=True),
        y_true - y_true.mean(dim=0, keepdim=True),
    )
    return 1 - pearson_loss.mean()

def mse_pearson(y_true, y_pred):
    mse = torch.nn.functional.mse_loss(y_true, y_pred)
    pearson_loss = pearson(y_true, y_pred)
    return mse + 0.5 * pearson_loss

def smooth_l1_pearson(y_true, y_pred):
    smooth_l1 = torch.nn.functional.smooth_l1_loss(y_true, y_pred)
    pearson_loss = pearson(y_true, y_pred)
    return smooth_l1 + 0.5 * pearson_loss

def mse_skip_init(y_true, y_pred):
    mse = torch.nn.functional.mse_loss(y_true[100:, :], y_pred[100:, :])
    return mse

def smooth_l1_skip_init(y_true, y_pred):
    smooth_l1 = torch.nn.functional.smooth_l1_loss(y_true[100:, :], y_pred[100:, :])
    return smooth_l1

loss_functions = {
    'mse': torch.nn.MSELoss(),
    'l1': torch.nn.L1Loss(),
    'smooth_l1': torch.nn.SmoothL1Loss(),
    'mse+pearson': mse_pearson,
    'smooth_l1+pearson': smooth_l1_pearson,
    'pearson': pearson,
    'mse_skip_init': mse_skip_init,
    'smooth_l1_skip_init': smooth_l1_skip_init,
}
