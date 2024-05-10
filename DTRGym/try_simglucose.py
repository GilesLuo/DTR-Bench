import DTRGym
import gymnasium as gym
from DTRGym.simglucose_env import RandomPatientEnv
from DTRGym.utils import DiscreteActionWrapper
from DTRBench.utils.misc import set_global_seed
import pandas as pd
import matplotlib.pyplot as plt

set_global_seed(0)
import warnings
warnings.filterwarnings("ignore")

simulation_time = 72*60
n_act  = 11
env = RandomPatientEnv(72*60,
                       sample_time=1,
                       start_time=0,
                       candidates=["adolescent#001"],
                       random_init_bg=False,
                       random_obs=False, random_meal=True,
                       missing_rate=0)
env = DiscreteActionWrapper(env, n_act)
env.reset(0)
step = 0
print(env.patient_name)
done = False
df1 = []
for step in range(simulation_time):
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    df1.append({"obs": obs, "meal": info["meal"], "time": info["time"]})
    if done:
        print("truncated" if truncated else "terminated")
        break
    print(f"Step: {step} obs: {obs}, meal: {info['meal']}, time: {info['time']}")
df1 = pd.DataFrame(df1)

plt.plot(df1["obs"])
df1["meal"].plot()
plt.show()