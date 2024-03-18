import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import ActorCritic, MLP
from typing import Union, List, Tuple, Optional, Callable, Sequence, Dict, Any
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
import torch
from torch import nn
import torch.nn.functional as F
from tianshou.data.batch import Batch

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]


class Net(nn.Module):
    def __init__(
            self,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]] = 0,
            hidden_sizes: Sequence[int] = (),
            norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
            norm_args: Optional[ArgsType] = None,
            activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
            act_args: Optional[ArgsType] = None,
            device: Union[str, int, torch.device] = "cpu",
            softmax: bool = False,
            concat: bool = False,
            num_atoms: int = 1,
            dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
            linear_layer: Type[nn.Linear] = nn.Linear,
            cat_num: int = 1,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.cat_num = cat_num
        input_dim = int(np.prod(state_shape)) * cat_num
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        if obs.ndim == 3:
            obs = obs.reshape(obs.shape[0], -1)
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Recurrent(nn.Module):
    def __init__(
            self,
            layer_num: int,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]],
            device: Union[str, int, torch.device] = "cpu",
            hidden_layer_size: int = 128,
            dropout: float = 0.0,
            num_atoms: int = 1,
            last_step_only: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            dropout=dropout,
            batch_first=True,
        )
        self.num_atoms = num_atoms
        self.action_dim = int(np.prod(action_shape)) * num_atoms
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, self.action_dim)
        self.use_last_step = last_step_only

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        if self.use_last_step:
            obs = self.fc2(obs[:, -1])
        else:
            obs = self.fc2(obs)

        if self.num_atoms > 1:
            obs = obs.view(obs.shape[0], -1, self.num_atoms)

        return obs, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }

class RecurrentPreprocess(nn.Module):
    def __init__(
            self,
            layer_num: int,
            state_shape: Union[int, Sequence[int]],
            device: Union[str, int, torch.device] = "cpu",
            hidden_layer_size: int = 128,
            dropout: float = 0.0,
            last_step_only: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            dropout=dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.use_last_step = last_step_only

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        if self.use_last_step:
            obs = obs[:, -1]
        # please ensure the first dim is batch size: [bsz, len, ...]
        return obs, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }

def define_single_network(input_shape: int, output_shape: int,
                          use_rnn=False, use_dueling=False, cat_num: int = 1, linear=False,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          ):
    if use_dueling and use_rnn:
        raise NotImplementedError("rnn and dueling are not implemented together")

    if use_dueling:
        if linear:
            dueling_params = ({"hidden_sizes": (), "activation": None},
                              {"hidden_sizes": (), "activation": None})
        else:
            dueling_params = ({"hidden_sizes": (256, 256), "activation": nn.ReLU},
                              {"hidden_sizes": (256, 256), "activation": nn.ReLU})
    else:
        dueling_params = None
    if not use_rnn:
        net = Net(state_shape=input_shape, action_shape=output_shape,
                  hidden_sizes=(256, 256, 256, 256) if not linear else (), activation=nn.ReLU if not linear else None,
                  device=device, dueling_param=dueling_params, cat_num=cat_num).to(device)
    else:
        net = Recurrent(layer_num=3,
                        state_shape=input_shape,
                        action_shape=output_shape,
                        device=device,
                        hidden_layer_size=256,
                        ).to(device)

    return net


class QRDQN(nn.Module):
    """Reference: Distributional Reinforcement Learning with Quantile \
    Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, state_shape, action_shape, hidden_sizes=(256, 256, 256, 256), activation=nn.ReLU,
                 num_quantiles=200, cat_num: int = 1, device="cpu"):
        super(QRDQN, self).__init__()
        self.input_shape = state_shape
        self.action_shape = action_shape
        self.cat_num = cat_num
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.num_quantiles = num_quantiles
        self.device = device
        model_list = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                model_list.append(nn.Linear(state_shape * self.cat_num, hidden_sizes[i]))
            else:
                model_list.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            model_list.append(self.activation())
        if hidden_sizes:
            model_list.append(nn.Linear(hidden_sizes[-1], action_shape * num_quantiles))
        else:
            model_list.append(nn.Linear(state_shape * self.cat_num, action_shape * num_quantiles))
        self.model = nn.Sequential(*model_list)

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs.ndim == 3:
            obs = obs.reshape(obs.shape[0], -1)
        obs = self.model(obs)
        obs = obs.view(-1, self.action_shape, self.num_quantiles)
        return obs, state


class SeqEncoder(nn.Module):
    def __init__(self, backbone_num_layer, feature_dim, hidden_size,
                 device: Union[str, int, torch.device] = "cpu"):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_size).to(device)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=backbone_num_layer,
            batch_first=True,
        ).to(device)

        self.device = device

    def forward(self, x: torch.Tensor, state=None, info={}):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        x = self.fc1(x)
        self.lstm.flatten_parameters()
        if state is None:
            x, (hidden, cell) = self.lstm(x)
        else:
            x, (hidden, cell) = self.lstm(
                x, (
                    state["encoder_hidden"].transpose(0, 1).contiguous(),
                    state["encoder_cell"].transpose(0, 1).contiguous()
                )
            )
        return x, {
            "encoder_hidden": hidden.transpose(0, 1).detach(),
            "encoder_cell": cell.transpose(0, 1).detach()
        }


class SSLNet(nn.Module):
    def __init__(self, backbone_num_layer, feature_dim, hidden_size,
                 device: Union[str, int, torch.device] = "cpu"):
        super().__init__()
        self.encoder = SeqEncoder(backbone_num_layer, feature_dim, hidden_size, device)
        self.decoder = SeqEncoder(backbone_num_layer, hidden_size, feature_dim, device)
        self.device = device

    def forward(self, x: torch.Tensor, state=None, info={}):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if state is None:
            x, e_state = self.encoder(x)
            x, d_state = self.decoder(x)
        else:
            x, e_state = self.encoder(
                x, (
                    state["encoder_state"]["hidden"].transpose(0, 1).contiguous(),
                    state["encoder_state"]["cell"].transpose(0, 1).contiguous()
                )
            )
            x, d_state = self.decoder(
                x, (
                    state["decoder_state"]["hidden"].transpose(0, 1).contiguous(),
                    state["decoder_state"]["cell"].transpose(0, 1).contiguous()
                )
            )
        encoder_state = {
            "hidden": e_state["hidden"].transpose(0, 1).detach(),
            "cell": e_state["cell"].transpose(0, 1).detach()
        }
        decoder_state = {
            "hidden": d_state["hidden"].transpose(0, 1).detach(),
            "cell": d_state["cell"].transpose(0, 1).detach()
        }

        return x, {"encoder_state": encoder_state, "decoder_state": decoder_state}


class PredLOSNet(nn.Module):
    def __init__(self, backbone_num_layer, task_num_layer, feature_dim, output_dim, hidden_size,
                 activation=nn.ReLU, norm_layer=None, episodic=False,
                 device: Union[str, int, torch.device] = "cpu"):
        super().__init__()
        self.backbone = SeqEncoder(backbone_num_layer, feature_dim, hidden_size, device).to(device)
        self.episodic = episodic
        self.pred_los_net = MLP(input_dim=hidden_size, output_dim=output_dim,
                                hidden_sizes=[hidden_size] * task_num_layer,
                                activation=activation, norm_layer=norm_layer, device=device, flatten_input=False).to(
            device)
        self.device = device

    def forward(self, x: torch.Tensor, state=None, info={}):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float)
        x, backbone_state = self.backbone(x)
        if not self.episodic:
            x = x[:, -1, :]
        LOS = self.pred_los_net(x)
        return LOS, None


if __name__ == "__main__":
    input_shape = 3
    output_shape = 2
    network = define_single_network(input_shape, output_shape,
                                    use_rnn=False, use_dueling=False, cat_num=2, linear=True)
    print(network)
    network = define_single_network(input_shape, output_shape,
                                    use_rnn=False, use_dueling=False, cat_num=1, linear=True)
    print(network)
