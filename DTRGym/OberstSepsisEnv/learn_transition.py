import itertools as it
import os

import numpy as np
from DTRGym.OberstSepsisEnv.state import State
from DTRGym.OberstSepsisEnv.action import Action
from DTRGym.OberstSepsisEnv.MDP import MDP
import cf.counterfactual as cf
import pickle as pkl
import ray
from tqdm import tqdm


def generate_mdp_matrices(n_iter, num_cpus, save_path):
    # Initialize Ray if num_cpus is more than 1
    if num_cpus > 1:
        ray.init(num_cpus=num_cpus)

    # Samples per component/state/action pair
    np.random.seed(1)
    n_actions = Action.NUM_ACTIONS_TOTAL
    n_states = State.NUM_OBS_STATES
    n_components = 2

    states = range(n_states)
    actions = range(n_actions)
    components = [0, 1]

    ## TRANSITION MATRIX
    tx_mat = np.zeros((n_components, n_actions, n_states, n_states))  # [has_diabetes, action, s_{t}, s_{t+1}]

    # Not used, but a required argument
    dummy_pol = np.ones((n_states, n_actions)) / n_actions

    if num_cpus != 1:
        import warnings
        warnings.warn("num_cpus=1 is usually faster!.")

        # Define the simulation function
        @ray.remote(num_cpus=num_cpus)
        def simulate(c, s0, a):
            this_mdp = MDP(init_state_idx=s0, policy_array=dummy_pol, p_diabetes=c)
            r = this_mdp.transition(Action(action_idx=a))
            s1 = this_mdp.state.get_state_idx()
            return (c, a, s0, s1)

        # Start all simulations
        futures = [simulate.remote(c, s0, a) for (c, s0, a, _) in
                   tqdm(it.product(components, states, actions, range(n_iter)),
                        total=n_components * n_actions * n_states * n_iter, desc="Submitting jobs")]

        # Collect results
        pbar = tqdm(total=n_components * n_actions * n_states * n_iter, desc="Completed jobs")
        while futures:
            done, futures = ray.wait(futures)
            for future in done:
                c, a, s0, s1 = ray.get(future)
                tx_mat[c, a, s0, s1] += 1
                pbar.update(1)
        pbar.close()
    else:
        # Run simulations sequentially
        for (c, s0, a, _) in tqdm(it.product(components, states, actions, range(n_iter)),
                                  total=n_components * n_actions * n_states * n_iter, desc="Simulations"):
            this_mdp = MDP(init_state_idx=s0, policy_array=dummy_pol, p_diabetes=c)
            r = this_mdp.transition(Action(action_idx=a))
            s1 = this_mdp.state.get_state_idx()
            tx_mat[c, a, s0, s1] += 1

    est_tx_mat = tx_mat / n_iter
    # Extra normalization
    est_tx_mat /= est_tx_mat.sum(axis=-1, keepdims=True)

    print("Transition matrix generated.")

    ## REWARD MATRIX
    np.random.seed(1)

    # Calculate the reward matrix explicitly, only based on state
    est_r_mat = np.zeros_like(est_tx_mat)
    for s1 in states:
        this_mdp = MDP(init_state_idx=s1, policy_array=dummy_pol, p_diabetes=1)  # p_diabetes does not matter here
        r = this_mdp.calculateReward()
        est_r_mat[:, :, :, s1] = r

    ## PRIOR ON INITIAL STATE
    np.random.seed(1)
    prior_initial_state = np.zeros((n_components, n_states))

    for c in components:
        this_mdp = MDP(p_diabetes=c)
        for _ in range(n_iter):
            s = this_mdp.get_new_state().get_state_idx()
            prior_initial_state[c, s] += 1

    prior_initial_state = prior_initial_state / n_iter
    # Extra normalization
    prior_initial_state /= prior_initial_state.sum(axis=-1, keepdims=True)

    # Save the matrices
    mat_dict = {"tx_mat": est_tx_mat,
                "r_mat": est_r_mat}
    np.savez_compressed(save_path, **mat_dict)

    print("Matrices saved to ", os.path.abspath(save_path))

    return mat_dict


def generate_optimal_policy(mdp_file, gamma=0.99):
    n_actions = Action.NUM_ACTIONS_TOTAL
    # load data
    # Get the transition and reward matrix from file
    mdict = np.load(mdp_file)
    if "fullPol" in mdict and mdict["fullPol"] is not None:
        print("Policy already exists in file. Skipping.")
        return mdict["fullPol"], mdict["tx_mat"], mdict["r_mat"]
    tx_mat = mdict["tx_mat"]
    r_mat = mdict["r_mat"]

    from scipy.linalg import block_diag

    tx_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))
    r_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))

    for a in range(n_actions):
        tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a, ...])
        r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])

    fullMDP = cf.MatrixMDP(tx_mat_full, r_mat_full)
    print("Solving MDP...")
    fullPol = fullMDP.policyIteration(discount=gamma, eval_type=1)
    new_dict = {
        "fullPol": fullPol,
        "tx_mat": tx_mat,
        "r_mat": r_mat,
    }
    np.savez_compressed(mdp_file, **new_dict)
    print("Optimal policy saved to ", os.path.abspath(mdp_file))
    return fullPol, tx_mat, r_mat


def main(save_path, n_iter, num_cpus=1, gamma=0.99):
    if not os.path.exists(save_path):
        print(save_path, "does not exist. Generating MDP matrices...")
        generate_mdp_matrices(n_iter=n_iter, num_cpus=num_cpus, save_path=save_path)
    return generate_optimal_policy(save_path, gamma=gamma)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generates MDP matrices for Oberst et al. sepsis model.')
    parser.add_argument('--n_iter', type=int, default=1000,
                        help='Number of simulations to run for each combination of components, actions, and states.')
    parser.add_argument('--num_cpus', type=int, default=1,
                        help='Number of CPU cores to use for parallelization.')
    parser.add_argument('--save_path', type=str, default="./mdp_matrices.npz",
                        help='Path to save the MDP matrices to.')
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor for optimal policy generation.")
    args = parser.parse_args()
    print("File will be saved to ", os.path.abspath(args.save_path))
    main(args.save_path, args.n_iter, args.num_cpus)
