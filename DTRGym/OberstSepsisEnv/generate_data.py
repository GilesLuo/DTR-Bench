# generate offline data by empirical transition matrix, convert to continuous data, and then save it to tianshou buffer
# most code copied from https://github.com/clinicalml/gumbel-max-scm/blob/sim-v2/plots-main-paper.ipynb
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts, dim
import DTRGym.OberstSepsisEnv.cf.counterfactual as cf
import cf.utils as utils
from tqdm.auto import tqdm

from DTRGym.OberstSepsisEnv.state import State
from DTRGym.OberstSepsisEnv.action import Action
from DTRGym.OberstSepsisEnv.DataGenerator import DataGenerator
from DTRGym.OberstSepsisEnv.env import SimSepsisEnv
from DTRGym.OberstSepsisEnv.learn_transition import main as get_full_policy
import warnings
from bokeh.io import export_png
import h5py
from pathlib import Path

hv.extension('bokeh')
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def set_logger(filename="./data generation.log"):
    sys.stdout = Logger(filename)


# This is a QA function to ensure that the RL policy is only taking actions that have been observed
def check_rl_policy(rl_policy, obs_samps, proj_lookup, r_mat, max_len, num_traj):
    passes = True
    # Check the observed actions for each state
    obs_pol = np.zeros_like(rl_policy)
    for eps_idx in range(num_traj):
        for time_idx in range(max_len):
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])
            # Need to get projected state
            if this_obs_action == -1:
                continue
            this_obs_state = proj_lookup[int(obs_samps[eps_idx, time_idx, 2])]
            obs_pol[this_obs_state, this_obs_action] += 1

    # Check if each RL action conforms to an observed action
    for eps_idx in range(num_traj):
        for time_idx in range(max_len):
            this_full_state_unobserved = int(obs_samps[eps_idx, time_idx, 1])
            this_obs_state = proj_lookup[this_full_state_unobserved]
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])

            if this_obs_action == -1:
                continue
            # This is key: In some of these trajectories, you die or get discharge.
            # In this case, no action is taken because the sequence has terminated, so there's nothing to compare the RL action to
            true_death_states = r_mat[0, 0, 0, :] == -1
            true_disch_states = r_mat[0, 0, 0, :] == 1
            if np.logical_or(true_death_states, true_disch_states)[this_full_state_unobserved]:
                continue

            this_rl_action = rl_policy[proj_lookup[this_obs_state]].argmax()
            if obs_pol[this_obs_state, this_rl_action] == 0:
                print("Eps: {} \t RL Action {} in State {} never observed".format(
                    int(time_idx / max_len), this_rl_action, this_obs_state))
                passes = False
    return passes


def projection_func_wrapper(proj_lookup):
    def projection_func(obs_state_idx):
        if obs_state_idx == -1:
            return -1
        else:
            return proj_lookup[obs_state_idx]

    return projection_func


def filter_trajectories(obs_samps, repetition_threshold):
    num_traj, max_len, _ = obs_samps.shape
    actions = obs_samps[:, :, 1]

    valid_idx = []
    rep_prop = []
    for i in range(num_traj):
        traj_actions = actions[i]

        # Create mask of where repetitions occur
        repetition_mask = np.zeros_like(traj_actions)
        repetition_mask[1:] = (traj_actions[1:] == traj_actions[:-1]) & (traj_actions[:-1] != -1)

        # Calculate proportion of repeated actions
        repetition_proportion = np.sum(repetition_mask) / (np.sum(traj_actions != -1) - 1) if np.sum(
            traj_actions != -1) > 1 else 0
        rep_prop.append(repetition_proportion)
        if repetition_proportion >= repetition_threshold:
            valid_idx.append(i)

    return obs_samps[valid_idx, :, :], rep_prop


def generate_bc_trajectories(num_traj, max_len, gamma, phys_epsilon, prob_diab, bootstrap_pec, repetition_threshold):
    max_len = max_len  # Max length of each trajectory
    gamma = gamma  # Used for computing optimal policies
    DISCOUNT = 1  # Used for computing actual reward

    # Option 1: Use bootstrapping w/replacement on the original num_traj to estimate errors
    USE_BOOSTRAP = True
    N_BOOTSTRAP = int(num_traj * bootstrap_pec)
    # Option 2: Use repeated sampling (i.e., num_traj fresh simulations each time) to get error bars;
    # This is done in the appendix of the paper, but not in the main paper
    N_REPEAT_SAMPLING = 1

    # These are properties of the simulator, do not change
    n_actions = Action.NUM_ACTIONS_TOTAL

    # These are added as absorbing states
    n_states_abs = State.NUM_OBS_STATES + 2
    discStateIdx = n_states_abs - 1
    deadStateIdx = n_states_abs - 2

    # load data
    # Get the transition and reward matrix from file

    fullPol, tx_mat, r_mat = get_full_policy(n_iter=1000, gamma=gamma,
                                             save_path=str(Path(__file__).resolve().parent / "mdp_matrices.npz"))
    physPolSoft = np.copy(fullPol)
    physPolSoft[physPolSoft == 1] = 1 - phys_epsilon
    physPolSoft[physPolSoft == 0] = phys_epsilon / (n_actions - 1)

    all_samps = np.array([])
    all_rep = np.array([])

    # Construct the projection matrix for obs->proj states
    n_proj_states = int((n_states_abs - 2) / 5) + 2
    proj_matrix = np.zeros((n_states_abs, n_proj_states))
    for i in range(n_states_abs - 2):
        this_state = State(state_idx=i, idx_type='obs',
                           diabetic_idx=1)  # Diab a req argument, no difference
        # assert this_state == State(state_idx = i, idx_type = 'obs', diabetic_idx = 0)
        j = this_state.get_state_idx('proj_obs')
        proj_matrix[i, j] = 1

    # Add the projection to death and discharge
    proj_matrix[deadStateIdx, -2] = 1
    proj_matrix[discStateIdx, -1] = 1

    proj_matrix = proj_matrix.astype(int)

    proj_lookup = proj_matrix.argmax(axis=-1)

    for it in tqdm(range(N_REPEAT_SAMPLING), desc="Outer Loop"):
        np.random.seed(it)
        done_traj = 0
        pbar = tqdm(total=num_traj, desc=f"generating traj with seed {it}-gamma {phys_epsilon} ")
        while len(all_samps) < num_traj:
            dgen = DataGenerator()
            states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals = dgen.simulate(
                int((num_traj - len(all_samps)) * 1.4), max_len, policy=physPolSoft, policy_idx_type='full',
                p_diabetes=prob_diab, use_tqdm=False)  # True, tqdm_desc='Behaviour Policy Simulation')

            obs_samps = utils.format_dgen_samps(
                states, actions, rewards, diab, max_len, int((num_traj - len(all_samps)) * 1.4))
            obs_samps, action_rep = filter_trajectories(obs_samps, repetition_threshold)
            obs_samps = obs_samps[:num_traj - len(all_samps), :, :]
            pbar.update(obs_samps.shape[0])

            if len(all_samps) == 0:
                all_samps = obs_samps
                all_rep = action_rep
            else:
                all_samps = np.concatenate((all_samps, obs_samps), axis=0)
                all_rep = np.concatenate((all_rep, action_rep), axis=0)
    return all_samps, all_rep


def save_data(all_samples, buffer_save_path, dataset_save_path, gamma, noise):
    """
    Save the samples to a tianshou buffer
    """

    from tianshou.data import ReplayBuffer
    from tianshou.data import Batch
    all_batch = []

    for episode_idx in tqdm(range(len(all_samples)), desc='saving data to batch {}'.format(buffer_save_path)):
        ep_batch = []
        episode_data = all_samples[episode_idx]
        LOS = (episode_data[:, 1] != -1).sum().astype(int)

        for step in range(LOS):  # action is not -1
            _, action, state_idx, next_state_idx, _, _, r = episode_data[step, :]
            if step == 0:
                act_rep = 0
            else:
                last_action = all_samples[episode_idx, step - 1, 1]
                act_rep = int(last_action == action)

            # convert state_idx to state

            s = SimSepsisEnv.state_idx_decompose(state_idx, idx_type='full')
            proj_s = s[1:4] + s[5:]
            if step == 0:
                o = list(SimSepsisEnv.discrete2conti(*s[1:5], noise=noise))  # take projected obs
                o = np.array(o[:-1] + s[5:])  # remove glucose, add intervention
            else:
                o = o_.copy()  # last obs_next
            s_ = SimSepsisEnv.state_idx_decompose(next_state_idx, idx_type='full')
            o_ = list(SimSepsisEnv.discrete2conti(*s_[1:5], noise=noise))
            o_ = np.array(o_[:-1] + s_[5:])

            terminated = r != 0
            truncated = step == LOS - 1 and r == 0
            assert not (terminated and truncated)
            ep_batch.append(
                Batch(
                    obs=o.astype(float),
                    act=action,
                    rew=r * 100,
                    terminated=terminated,
                    truncated=truncated,
                    done=terminated or truncated,
                    obs_next=o_.astype(float),
                    info={
                        "episode_id": episode_idx,
                        "step": step,
                        "LOS": np.array(LOS - step - 1, dtype=float).reshape(-1),
                        "cur_act_idx": action,
                        "cur_act": SimSepsisEnv.action_idx_decompose(action),
                        "full_state": s,
                        "proj_state": proj_s,
                        "Q": gamma ** (LOS - step - 2) * r * 100,
                        "act_rep": act_rep,
                        "survived": r != -1,
                    }
                )

            )
            step += 1
        ep_batch = Batch.stack(ep_batch)
        assert (ep_batch.obs[1:, :] == ep_batch.obs_next[:-1, :]).all()
        all_batch.append(Batch.stack(ep_batch))
    print("collating all batches, this may take a while")
    all_batch = Batch.cat(all_batch)

    print("saving to hdf5")
    with h5py.File(dataset_save_path, 'w') as f:
        for key in all_batch.keys():
            if key != "info":
                f.create_dataset(key, data=all_batch[key])  # hdf5 cannot save dict, so we save it separately
        # save batch.info again separately for easy access
        for key in all_batch.info.keys():
            if key in f:
                raise ValueError("key {} already exists in hdf5 file. This is because the key in"
                                 " info and key in batch have a conflict".format(key))
            f.create_dataset(key, data=all_batch.info[key])

    print("saving to buffer")
    replay_buffer = ReplayBuffer(size=all_batch.shape[0])
    replay_buffer.set_batch(all_batch)
    replay_buffer.save_hdf5(buffer_save_path)
    return replay_buffer


def save_chord_to_png(chord, filename):
    # Render the Chord diagram as a Bokeh plot
    plot = hv.render(chord)

    # Save the Bokeh plot to a PNG file
    export_png(plot, filename=filename)


def create_state_transition_matrix(obs_samps):
    states = obs_samps[:, :, 2:4]

    # remove the state where both dim are -1
    states = states[np.logical_and(states[:, :, 0] != -1, states[:, :, 1] != -1)]
    states_from, states_to = states[:, 0], states[:, 1]
    n_states = len(states)
    # Create a 2D histogram of state transitions
    hist, xedges, yedges = np.histogram2d(
        states_from.flatten(), states_to.flatten(), bins=np.unique(np.concatenate([states_from, states_to])))

    # Normalize to get prevalence
    hist /= np.sum(hist)

    return hist


def create_chord_data(matrix):
    # Create a DataFrame for the Chord diagram
    data = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                data.append((str(i), str(j), matrix[i, j]))
    return pd.DataFrame(data, columns=['source', 'target', 'value'])


def draw_chord_diagram(data):
    # Define the Chord diagram
    chord = hv.Chord(data)

    # Set options
    chord.opts(
        opts.Chord(cmap='Category10', edge_cmap='Category10', edge_color=dim('source').str(),
                   labels='index', node_color=dim('index').str())
    )
    return chord


def main():
    import argparse

    import numpy as np
    import random
    # Absolute path to the script file
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    np.random.seed(0)
    random.seed(0)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--num_traj', type=int, default=5000)
    argparser.add_argument('--max_len', type=int, default=40)
    argparser.add_argument('--gamma', type=float, default=0.95)
    argparser.add_argument('--epsilon', type=float, nargs='*', default=[0, 0.1])
    argparser.add_argument('--noise', type=float, default=0.2)
    argparser.add_argument('--save_dir', type=str, default=Path(script_dir) / "offline_data")
    argparser.add_argument('--repetition_threshold', nargs='*', default=[0, 0.5, 0.75], type=int)
    argparser.add_argument('--plot_state_chord', action='store_true')
    argparser.add_argument('--plot_action_repeat', action='store_true')
    args = argparser.parse_args()

    set_logger(args.save_dir / "generate_bc_trajectories.log")

    for ep in args.epsilon:
        for thresh in args.repetition_threshold:
            name_ext = "{}_ep{}_rep{}".format(args.num_traj, ep, thresh)
            print("generating data for {}".format(name_ext))
            os.makedirs(args.save_dir, exist_ok=True)
            all_samps, all_rep = generate_bc_trajectories(args.num_traj, args.max_len, args.gamma, ep, 0.2, 0.1,
                                                          repetition_threshold=thresh)
            print(f"biased_buffer_{name_ext} behaviroal reward: {all_samps[:, :, 6].sum() / args.num_traj}")
            if args.plot_state_chord:
                state_hist = create_state_transition_matrix(all_samps)
                chord_data = create_chord_data(state_hist)
                chord = draw_chord_diagram(chord_data)
                save_chord_to_png(chord, Path(args.save_dir) / f"state_chord_{name_ext}.png")

            if args.plot_action_repeat:
                plt.hist(all_rep, bins=20)
                plt.savefig(Path(args.save_dir) / f"action_rep_{name_ext}.png")

            save_data(all_samps,
                      Path(args.save_dir) / f"biased_buffer_{name_ext}.hdf5",
                      Path(args.save_dir) / f"biased_dataset_{name_ext}.hdf5",
                      gamma=args.gamma, noise=args.noise)


if __name__ == "__main__":
    main()
