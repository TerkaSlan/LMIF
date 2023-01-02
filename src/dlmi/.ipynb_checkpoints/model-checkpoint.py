import torch
from torch import nn
import torch.nn.functional as nnf
from dlmi.utils import get_device, reverse_dict
import numpy as np


DEVICE = get_device()


class Model(nn.Module):
    def __init__(self, input_dim=4096, output_dim=10):
        super().__init__()
        self.layers = torch.nn.Sequential(
          torch.nn.Linear(input_dim, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, output_dim)
        )
        self.n_output_neurons = self.layers[-1].weight.shape[0]
        self.output_neurons = {
            i: j for i, j in zip(
                [i for i in range(self.n_output_neurons)],
                [i for i in range(self.n_output_neurons)]
            )
        }

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.layers(x)
        return outputs

    def add_units(self, n_output_neurons: int):
        m = self.layers[-1]
        old_shape = m.weight.shape

        m2 = nn.Linear(old_shape[1], old_shape[0] + n_output_neurons)
        m2.weight = nn.parameter.Parameter(torch.cat((m.weight, m2.weight[0:n_output_neurons])))
        m2.bias = nn.parameter.Parameter(torch.cat((m.bias, m2.bias[0:n_output_neurons])))
        self.layers[-1] = m2
        self.n_output_neurons = self.layers[-1].weight.shape[0]

    def remove_unit(self, unit: int):
        m = self.layers[-1]
        old_shape = m.weight.shape
        m2 = nn.Linear(old_shape[1], old_shape[0] - 1)
        unit_index = reverse_dict(self.output_neurons)[unit]
        m2.weight = nn.parameter.Parameter(torch.cat((m.weight[0:unit_index], m.weight[unit_index+1:old_shape[0]])))
        m2.bias = nn.parameter.Parameter(torch.cat((m.bias[0:unit_index], m.bias[unit_index+1:old_shape[0]])))
        self.layers[-1] = m2
        self.n_output_neurons = self.layers[-1].weight.shape[0]
        del self.output_neurons[unit_index]

        rest_outputs = list(self.output_neurons.values())
        self.output_neurons = {
            i: j for i, j in zip(
                [i for i in range(len(rest_outputs))],
                rest_outputs
            )
        }


class NeuralNetwork():
    def __init__(
        self,
        input_dim,
        output_dim,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        lr=0.001
    ):
        self.device = DEVICE
        self.model = Model(input_dim, output_dim).to(self.device)
        self.loss = loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, data_X: torch.FloatTensor):
        """ Collects predictions for multiple data points (used in structure building)."""
        self.model = self.model.to(self.device)
        self.model.eval()

        all_outputs = torch.tensor([], device=self.device)
        with torch.no_grad():
            outputs = self.model(data_X.to(self.device))
            all_outputs = torch.cat((all_outputs, outputs), 0)

        _, y_pred = torch.max(all_outputs, 1)
        return np.array([self.model.output_neurons[label] for label in y_pred.cpu().numpy()])

    def predict_single(self, data_X: torch.FloatTensor):
        """ Collects predictions for a single data point (used in query predictions)."""
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(data_X.to(self.device))

        prob = nnf.softmax(outputs, dim=0)
        top_prob, top_class = prob.topk(self.model.n_output_neurons, dim=0)
        top_prob = top_prob.cpu().numpy()
        return top_prob, np.array([self.model.output_neurons[label] for label in top_class.cpu().numpy()])

    def train(self, data_X: torch.FloatTensor, data_y: torch.LongTensor, epochs=50):
        losses = []
        for _ in range(epochs):
            pred_y = self.model(data_X.to(self.device))
            curr_loss = self.loss(pred_y, data_y.to(self.device))
            losses.append(curr_loss.item())

            self.model.zero_grad()
            curr_loss.backward()

            self.optimizer.step()
