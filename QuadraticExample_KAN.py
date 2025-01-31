from datetime import datetime

import matplotlib.pyplot as plt
import torch
import numpy as np

from FunctionEncoder import QuadraticDataset, FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, \
    DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
parser.add_argument("--hidden_size", type=int, default=10)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--activation", type=str, default="silu")
parser.add_argument("--grid_low", type=float, default=-10.0)  
parser.add_argument("--grid_high", type=float, default=10.0)
parser.add_argument("--grid_size", type=int, default=5)
parser.add_argument("--spline_order", type=int, default=3)

args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals
hidden_size = args.hidden_size
n_layers = args.n_layers
activation = args.activation
grid_range = [args.grid_low, args.grid_high]
grid_size = args.grid_size
spline_order = args.spline_order

if load_path is None:
    logdir = f"logs/quadratic_example/{train_method}/{'KAN'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "KAN"

# seed torch
torch.manual_seed(seed)

# create a dataset
if residuals:
    a_range = (0, 3/50) # this makes the true average function non-zero
else:
    a_range = (-3/50, 3/50)
b_range = (-3/50, 3/50)
c_range = (-3/50, 3/50)
input_range = (-10, 10)
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range, n_examples=100, n_queries=1000)

if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            model_kwargs = {"hidden_size": hidden_size, "n_layers": n_layers, "activation": activation, "grid_range":grid_range, "grid_size": grid_size, "spline_order": spline_order},
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    if train_method == "inner_product":
        y_hats_ip = model.predict_from_examples(example_xs, example_ys, query_xs, method="inner_product")
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")
    query_xs, indicies = torch.sort(query_xs, dim=-2)
    query_ys = query_ys.gather(dim=-2, index=indicies)
    y_hats_ls = y_hats_ls.gather(dim=-2, index=indicies)
    if train_method == "inner_product":
        y_hats_ip = y_hats_ip.gather(dim=-2, index=indicies)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(query_xs[i].cpu(), query_ys[i].cpu(), label="True")
        ax.plot(query_xs[i].cpu(), y_hats_ls[i].cpu(), label="LS")
        if train_method == "inner_product":
            ax.plot(query_xs[i].cpu(), y_hats_ip[i].cpu(), label="IP")
        if i == n_plots - 1:
            ax.legend()
        title = f"${info['As'][i].item():.2f}x^2 + {info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
        ax.set_title(title)
        y_min, y_max = query_ys[i].min().item(), query_ys[i].max().item()
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()

    # plot the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    query_xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)
    basis = model.forward_basis_functions(query_xs)
    for i in range(n_basis):
        ax.plot(query_xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
    if residuals:
        avg_function = model.average_function.forward(query_xs)
        ax.plot(query_xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

    plt.tight_layout()
    plt.savefig(f"{logdir}/basis.png")

# evaluate model
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range, n_functions=2048, n_examples=100, n_queries=1000)
with torch.no_grad():
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()
    if train_method == "inner_product":
        y_hats = model.predict_from_examples(example_xs, example_ys, query_xs, method="inner_product")
    y_hats = model.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")
    query_ys = query_ys.squeeze(-1).cpu().numpy()
    y_hats   = y_hats.squeeze(-1).cpu().numpy()
    
    relative_errors = np.linalg.norm(y_hats - query_ys, axis=1) / np.linalg.norm(query_ys, axis=1)
    print('L2 Relative Error: ', np.mean(relative_errors))
