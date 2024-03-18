from tianshou.policy import TD3Policy, SACPolicy, C51Policy, DiscreteSACPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.policy.modelfree.ddpg import DDPGPolicy
import torch
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.utils.net.discrete import Actor as discreteActor
from tianshou.utils.net.discrete import Critic as discreteCritic

from DTRBench.utils.network import define_single_network, Net, QRDQN, RecurrentPreprocess, Recurrent


def get_DQN(obs_shape, action_shape, weights_path=None):
    model = define_single_network(input_shape=obs_shape, output_shape=action_shape)
    policy = DQNPolicy(model=model, optim=None)
    # only keeps the model. prefix in the state_dict
    if weights_path is not None:
        original_state_dict = torch.load(weights_path, map_location="cpu")
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
        policy.load_state_dict(new_state_dict)

    policy.eval()
    return policy


def get_DDQN(obs_shape, action_shape, weights_path=None):
    model = define_single_network(input_shape=obs_shape, output_shape=action_shape)
    policy = DQNPolicy(model=model, optim=None, is_double=True)

    # only keeps the model. prefix in the state_dict
    if weights_path is not None:
        original_state_dict = torch.load(weights_path, map_location="cpu")
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
        policy.load_state_dict(new_state_dict)

    policy.eval()
    return policy


def get_DDQN_dueling(obs_shape, action_shape, weights_path=None):
    model = define_single_network(input_shape=obs_shape, output_shape=action_shape,
                                  use_dueling=True)
    policy = DQNPolicy(model=model, optim=None, is_double=True)

    # only keeps the model. prefix in the state_dict
    if weights_path is not None:
        original_state_dict = torch.load(weights_path, map_location="cpu")
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
        policy.load_state_dict(new_state_dict)

    policy.eval()
    return policy


def get_C51(obs_shape, action_shape, weights_path=None):
    hidden_sizes = [256, 256, 256, 256]
    model = Net(obs_shape,
                action_shape,
                hidden_sizes,
                softmax=True,
                num_atoms=51)
    policy = C51Policy(model=model, optim=None)

    # only keeps the model. prefix in the state_dict
    if weights_path is not None:
        original_state_dict = torch.load(weights_path, map_location="cpu")
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
            elif key.startswith("support"):
                new_state_dict[key] = value

        policy.load_state_dict(new_state_dict)

    policy.eval()
    return policy


def get_discrete_SAC(obs_shape, action_shape, weights_path=None):
    hidden_sizes = [256, 256, 256]
    net_actor = Net(obs_shape, hidden_sizes=hidden_sizes)
    net_critic1 = Net(obs_shape, hidden_sizes=hidden_sizes)
    net_critic2 = Net(obs_shape, hidden_sizes=hidden_sizes)

    actor = discreteActor(net_actor,
                          action_shape=action_shape,
                          softmax_output=False,
                          preprocess_net_output_dim=256)
    critic1 = discreteCritic(net_critic1,
                             last_size=action_shape,
                             preprocess_net_output_dim=256)
    critic2 = discreteCritic(net_critic2,
                             last_size=action_shape,
                             preprocess_net_output_dim=256)

    policy = DiscreteSACPolicy(actor=actor,
                               actor_optim=None,
                               critic1=critic1,
                               critic1_optim=None,
                               critic2=critic2,
                               critic2_optim=None)

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        policy.load_state_dict(state_dict)

    policy.eval()
    return policy


def get_DQN_rnn(obs_shape, action_shape, weights_path=None):
    model = define_single_network(input_shape=obs_shape,
                                  output_shape=action_shape,
                                  use_rnn=True)
    policy = DQNPolicy(model=model, optim=None)
    # only keeps the model. prefix in the state_dict
    if weights_path is not None:
        original_state_dict = torch.load(weights_path, map_location="cpu")
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
        policy.load_state_dict(new_state_dict)

    policy.eval()
    return policy


def get_DDQN_rnn(obs_shape, action_shape, weights_path=None):
    model = define_single_network(input_shape=obs_shape,
                                  output_shape=action_shape,
                                  use_rnn=True)
    policy = DQNPolicy(model=model, optim=None, is_double=True)

    # only keeps the model. prefix in the state_dict
    if weights_path is not None:
        original_state_dict = torch.load(weights_path, map_location="cpu")
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
        policy.load_state_dict(new_state_dict)

    policy.eval()
    return policy


def get_C51_rnn(obs_shape, action_shape, weights_path=None):
    model = Recurrent(layer_num=3, state_shape=obs_shape, action_shape=action_shape,
                      hidden_layer_size=256, num_atoms=51, last_step_only=True)

    policy = C51Policy(model=model, optim=None)

    # only keeps the model. prefix in the state_dict
    if weights_path is not None:
        original_state_dict = torch.load(weights_path, map_location="cpu")
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
            elif key.startswith("support"):
                new_state_dict[key] = value
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in policy.state_dict()}
        policy.load_state_dict(new_state_dict)

    policy.eval()
    return policy


def get_discrete_SAC_rnn(obs_shape, action_shape, weights_path=None):
    net_actor = RecurrentPreprocess(layer_num=3, state_shape=obs_shape, hidden_layer_size=256)
    net_critic1 = RecurrentPreprocess(layer_num=3, state_shape=obs_shape, hidden_layer_size=256)
    net_critic2 = RecurrentPreprocess(layer_num=3, state_shape=obs_shape, hidden_layer_size=256)

    actor = discreteActor(net_actor,
                          action_shape=action_shape,
                          softmax_output=False,
                          preprocess_net_output_dim=256)
    critic1 = discreteCritic(net_critic1,
                             last_size=action_shape,
                             preprocess_net_output_dim=256)
    critic2 = discreteCritic(net_critic2,
                             last_size=action_shape,
                             preprocess_net_output_dim=256)

    policy = DiscreteSACPolicy(actor=actor,
                               actor_optim=None,
                               critic1=critic1,
                               critic1_optim=None,
                               critic2=critic2,
                               critic2_optim=None)

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        policy.load_state_dict(state_dict)

    policy.eval()
    return policy


def get_DDPG(obs_shape, action_space, weights_path=None):
    max_action = action_space.high
    action_shape = action_space.shape

    hidden_sizes = [256, 256, 256, 256]
    net_a = Net(obs_shape, hidden_sizes=hidden_sizes)
    net_c = Net(
        obs_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    )

    actor = Actor(net_a, action_shape, max_action=torch.tensor(max_action),
                  preprocess_net_output_dim=256)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic = Critic(net_c)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    policy = DDPGPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        action_scaling=True,
        action_space=action_space)
    policy.eval()

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        policy.load_state_dict(state_dict)

    policy.eval()
    return policy


def get_TD3(obs_shape, action_space, weights_path=None):
    max_action = action_space.high[0]
    action_shape = action_space.shape
    hidden_sizes = [256, 256, 256, 256]

    net_a = Net(obs_shape, hidden_sizes=hidden_sizes)
    actor = Actor(net_a, action_shape, max_action=max_action)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    net_c1 = Net(
        obs_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True)
    net_c2 = Net(
        obs_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True)
    critic1 = Critic(net_c1)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2 = Critic(net_c2)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        action_scaling=True,
        action_space=action_space,
    )

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        policy.load_state_dict(state_dict)

    policy.eval()
    return policy


def get_SAC(obs_shape, action_space, weights_path=None):
    action_shape = action_space.shape
    hidden_sizes = [256, 256, 256]

    # model
    net_a = Net(obs_shape, hidden_sizes=hidden_sizes)
    actor = ActorProb(
        net_a,
        action_shape,
        unbounded=True,
        conditioned_sigma=True)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    net_c1 = Net(obs_shape, action_shape, hidden_sizes=hidden_sizes, concat=True, )
    net_c2 = Net(obs_shape, action_shape, hidden_sizes=hidden_sizes, concat=True, )
    critic1 = Critic(net_c1)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2 = Critic(net_c2)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        action_space=action_space,
    )

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        policy.load_state_dict(state_dict)

    policy.eval()
    return policy
