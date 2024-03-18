import shutil
import optuna
import pandas as pd
import os
from pathlib import Path

from DTRGym import envs


def detect_env(s: str) -> str:
    matched_envs = [env for env in envs if env in s]
    if len(matched_envs) > 1:
        raise ValueError(f"Multiple environments matched: {matched_envs}")
    elif len(matched_envs) == 0:
        raise ValueError("No environments matched")
    else:
        return matched_envs[0]


def summarize_test_result(root_dir):
    # Create an empty DataFrame with the required columns
    df = pd.DataFrame(columns=['env_name', 'action_type', 'algo_name', 'rew_mean', 'rew_std', 'len_mean'])

    # Navigate through the directories
    count = 0
    for env_dir in os.listdir(root_dir):
        algo_dir_path = os.path.join(root_dir, env_dir)
        if env_dir.startswith(".") or not os.path.isdir(algo_dir_path): continue  # Skip hidden files/directories
        env_name = detect_env(env_dir)
        action_type = "discrete" if "discrete" in env_dir else "continuous" if "continuous" in env_dir else None

        for algo_dir in os.listdir(algo_dir_path):
            seed_dir_path = os.path.join(algo_dir_path, algo_dir)
            if algo_dir.startswith(".") or "-best" not in algo_dir: continue
            algo_name = algo_dir.split('-best')[0]  # Remove '_best' from the end of algo name
            result = []
            for seed_dir in os.listdir(seed_dir_path):

                if seed_dir.startswith("."): continue
                csv_file_path = os.path.join(seed_dir_path, seed_dir, f'{env_dir}-{algo_name}-{seed_dir}.csv')
                # Read the csv file
                data = pd.read_csv(csv_file_path)
                # Compute mean and standard deviation of the reward
                result.append(data)
                # Append the data to the DataFrame
            data = pd.concat(result)
            rew_mean = data['rews'].mean()
            rew_std = data['rews'].std()
            len_mean = data['lens'].mean()

            df = pd.concat([df, pd.DataFrame([[env_dir, env_name, action_type, algo_name, rew_mean, rew_std, len_mean]],
                                             index=[count], columns=["study_name", 'env_name', 'action_type',
                                                                     'algo_name', 'rew_mean', 'rew_std', 'len_mean']
                                             )], ignore_index=True)
            count += 1
    return df


def summarize_db_result(log_dir, offline=False):
    """
    offline result is assumed to be val-offline-test-offline, so we can check everything in the .db file.
    check ALL .db files under the log_dir recursively and summarize the result
    """
    db_files = []
    studies = []

    # Step 1: Locate all .db files
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".db"):
                db_path = os.path.join(root, file)
                db_files.append([db_path, os.path.splitext(file)[0]])

    results = []
    for db_file, study_name in db_files:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_file}")
        trials_df = study.trials_dataframe()
        try:
            trials_df = trials_df[trials_df["state"] == "COMPLETE"]
            best_trial = trials_df.iloc[trials_df["value"].idxmax()]
        except Exception as e:
            print(f"Error in {db_file}: {e}")
            continue
        if offline:
            result = best_trial["user_attrs_result"]
            result = pd.Series(result)
            result["logdir"] = log_dir
            result["algo"] = best_trial["user_attrs_algo_name"]
        else:
            result = best_trial[[idx for idx in best_trial.index if idx.startswith("user_attrs_eval")]]
            result["logdir"] = log_dir
            result["algo"] = best_trial["user_attrs_algo_name"]
        result["best_trial"] = best_trial["number"] + 1
        result["completed_trials"] = len(study.trials)

        # parse study name
        study_name_list = study_name.split('-')
        if len(study_name_list) == 4:
            env_name, action_type, setting, algo_name= study_name_list[0], study_name_list[1], study_name_list[2], \
                                                                study_name_list[3]
        elif len(study_name_list) == 5:
            env_name, action_type, setting = study_name_list[0], study_name_list[1], study_name_list[2]
            algo_name = '-'.join(study_name_list[-2:])
        elif len(study_name_list) == 6:
            env_name, action_type, setting = study_name_list[0], study_name_list[1], study_name_list[2]
            algo_name = '-'.join(study_name_list[-3:])
        result["env_name"] = env_name
        result["algo_name"] = algo_name
        result["action_type"] = action_type
        result["setting"] = setting
        results.append(result)
    results = pd.concat(results, axis=1).T

    return results


def revise_test_result_fname(root_dir):
    # Navigate through the directories
    count = 0
    for env_dir in os.listdir(root_dir):
        algo_dir_path = os.path.join(root_dir, env_dir)
        if env_dir.startswith(".") or not os.path.isdir(algo_dir_path): continue  # Skip hidden files/directories
        env_name = detect_env(env_dir)

        for algo_dir in os.listdir(algo_dir_path):
            seed_dir_path = os.path.join(algo_dir_path, algo_dir)
            if algo_dir.startswith(".") or "-best" not in algo_dir: continue
            algo_name = algo_dir.split('-best')[0]  # Remove '_best' from the end of algo name
            for seed_dir in os.listdir(seed_dir_path):
                if seed_dir.startswith("."):
                    continue
                original_csv_fpath1 = os.path.join(seed_dir_path, seed_dir, 'test_result.csv')
                original_csv_fpath2 = os.path.join(seed_dir_path, seed_dir, f'{env_name}-{algo_name}-{seed_dir}.csv')
                if os.path.exists(original_csv_fpath1):
                    new_csv_name = f"{env_dir}-{algo_name}-{seed_dir}.csv"
                    new_csv_fpath = os.path.join(seed_dir_path, seed_dir, new_csv_name)
                    os.rename(original_csv_fpath1, new_csv_fpath)
                if os.path.exists(original_csv_fpath2):
                    new_csv_name = f"{env_dir}-{algo_name}-{seed_dir}.csv"
                    new_csv_fpath = os.path.join(seed_dir_path, seed_dir, new_csv_name)
                    os.rename(original_csv_fpath2, new_csv_fpath)

def collect_test_result_file(root_dir, collect_dir):
    Path(collect_dir).mkdir(parents=True, exist_ok=True)
    for env_dir in os.listdir(root_dir):
        algo_dir_path = os.path.join(root_dir, env_dir)
        if env_dir.startswith(".") or not os.path.isdir(algo_dir_path): continue  # Skip hidden files/directories

        for algo_dir in os.listdir(algo_dir_path):
            seed_dir_path = os.path.join(algo_dir_path, algo_dir)
            if algo_dir.startswith(".") or "-best" not in algo_dir: continue
            algo_name = algo_dir.split('-best')[0]  # Remove '_best' from the end of algo name
            for seed_dir in os.listdir(seed_dir_path):
                if seed_dir.startswith("."):
                    continue
                test_csv_fpath = os.path.join(seed_dir_path, seed_dir, f"{env_dir}-{algo_name}-{seed_dir}.csv")
                collected_csv_fpath = os.path.join(collect_dir, f"{env_dir}-{algo_name}-{seed_dir}.csv")
                if os.path.exists(test_csv_fpath):
                    shutil.copy(test_csv_fpath, collected_csv_fpath)


def summarize_test_results_from_collection(test_collection_dir):
    # Create an empty DataFrame with the required columns
    df = pd.DataFrame(columns=['env_name', 'action_type', 'algo_name', 'setting', 'rew_mean', 'rew_std', 'len_mean'])

    # Dictionary to store intermediate data
    intermediate_data = {}

    # Iterate through the files in the directory
    for test_csv_fpath in os.listdir(test_collection_dir):
        # Skip files starting with "."
        if test_csv_fpath.startswith("."):
            continue

        # Parse information from the file name
        test_csv_fname = os.path.splitext(test_csv_fpath)[0]
        test_fname_list = test_csv_fname.split('-')
        if len(test_fname_list) == 5:
            env_name, action_type, setting, algo_name, seed = test_fname_list[0], test_fname_list[1], test_fname_list[2], \
                                                                test_fname_list[3], test_fname_list[4]
        else:
            env_name, action_type, setting = test_fname_list[0], test_fname_list[1], test_fname_list[2]
            seed = test_fname_list[-1]
            algo_name = '-'.join(test_fname_list[3:-1])

        setting = int(setting[-1])

        # Read the CSV file
        file_path = os.path.join(test_collection_dir, test_csv_fpath)
        data = pd.read_csv(file_path)

        # Create a unique key for each combination excluding seed
        key = (env_name, action_type, algo_name, setting)

        # Aggregate data for each unique combination
        if key not in intermediate_data:
            intermediate_data[key] = {'rewards': [], 'lengths': []}

        intermediate_data[key]['rewards'].extend(data['rews'])
        intermediate_data[key]['lengths'].extend(data['lens'])

    # Calculate mean and std for each unique combination
    for key, values in intermediate_data.items():
        if len(values['rewards']) < 25000:
            print(f"Skipping {key} due to insufficient data")
            continue
        rew_mean = pd.Series(values['rewards']).mean()
        rew_std = pd.Series(values['rewards']).std()
        len_mean = pd.Series(values['lengths']).mean()

        # Append the aggregated data to the DataFrame
        df = df._append({'env_name': key[0],
                         'action_type': key[1],
                         'algo_name': key[2],
                         'setting': key[3],
                         'rew_mean': rew_mean,
                         'rew_std': rew_std,
                         'len_mean': len_mean},
                         ignore_index=True)

    return df


def generate_latex_tabular_content(csv_path: str, env_name: str, float_precision: int = 3, custom_order: list = None):
    df = pd.read_csv(csv_path)
    df = df[df['env_name'] == env_name]

    if custom_order is not None:
        df['algo_name'] = pd.Categorical(df['algo_name'], custom_order, ordered=True)

    df = df.sort_values(by=['algo_name', 'setting'])

    # Calculate max and second max rew_mean values for each setting across all algo_names
    max_values = {}
    second_max_values = {}
    settings = df['setting'].unique()
    for setting in settings:
        setting_df = df[df['setting'] == setting]
        max_values[setting] = setting_df['rew_mean'].max()
        if len(setting_df['rew_mean']) > 1:
            second_max_values[setting] = setting_df['rew_mean'].nlargest(2).iloc[-1]

    def generate_latex_table_row(algo_name, group, float_precision):
        reward_strs = []
        for setting in sorted(settings):
            if setting in group['setting'].values:
                row = group[group['setting'] == setting].iloc[0]
                rew_mean = row['rew_mean']
                rew_std = row['rew_std']

                color = ""
                if rew_mean == max_values[setting]:
                    color = "\\textcolor{red}{"
                elif rew_mean == second_max_values.get(setting, None):
                    color = "\\textcolor{blue}{"

                reward_str = f"{color}${rew_mean:.{float_precision}f} \\pm {rew_std:.{float_precision}f}${'}' if color else ''}"
            else:
                reward_str = "N/A"
            reward_strs.append(reward_str)

        latex_row = f"{algo_name} & {' & '.join(reward_strs)} \\\\"
        return latex_row

    # Generate LaTeX table rows for each algo_name
    latex_rows = [generate_latex_table_row(algo_name, group, float_precision) for algo_name, group in
                  df.groupby('algo_name')]

    latex_rows_str = '\n'.join(latex_rows)

    # Output the LaTeX code
    latex_code = "\\begin{tabular}{l|c c c c c} \n" \
                 "    \\hline\n" \
                 "    Policy & Setting 1 & Setting 2 & Setting 3 & Setting 4 & Setting 5 \\\\\n" \
                 "    \\hline\n" \
                 f"{latex_rows_str}\n" \
                 "    \\hline\n" \
                 "\\end{tabular}"

    return latex_code
