import numpy as np
import torch
from Env import QinshanEnv
from DDPG import ReplayBuffer, DDPG
import csv
import os
np.set_printoptions(linewidth=10_000)
def train_ddpg(
    env,
    agent,
    replay_buffer,
    num_episodes=200,
    batch_size=64,
    minimal_size=1000,
    gamma_update_every_step=True,
    noise_type="gaussian",
    noise_std=0.2,
    noise_decay=0.995,
    ou_theta=0.15,
    ou_sigma=0.2,
    save_every=10,
    device="gpu",
    reward_txt_path="./rewards.csv",
    power_dir="./power_logs",
    temp_dir="./temp_logs",
    model_dir="./models",
):
    os.makedirs(power_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # OU noise state (only when noise_type is "ou")
    ou_state = 0.0
    episode_rewards = []
    episode_scores = []

    best_score = float("inf")
    best_ep = None

    os.makedirs(model_dir, exist_ok=True)


    target_fracs = list(env.target_fractions)  # e.g. [0.9, 0.8, 0.7, 0.6]
    counts = {tf: 0 for tf in target_fracs}

    # Create the reward/score file and write the header once
    with open(reward_txt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "reward",
            "score",
            "mean_abs_error",
            "undershoot",
            "tail_mae",
            "settling_ratio"
        ])

    def get_noise():
        nonlocal ou_state, noise_std
        if noise_type == "gaussian":
            return np.random.normal(loc=0.0, scale=noise_std)
        elif noise_type == "ou":
            dx = ou_theta * (0.0 - ou_state) + ou_sigma * np.random.normal()
            ou_state = ou_state + dx
            return float(ou_state)
        else:
            return 0.0

    for ep in range(1, num_episodes + 1):

        # Random target per episode, preferring the least-trained condition
        min_count = min(counts[tf] for tf in target_fracs)
        candidates = [tf for tf in target_fracs if counts[tf] == min_count]
        chosen_tf = float(np.random.choice(candidates))

        obs, info = env.reset(options={"target_frac": chosen_tf})
        counts[chosen_tf] += 1

        print("Target power:", info["P_ref"])

        rho_accum = float(env.rho_accum)  # usually 0.0 right after env.reset

        ep_reward = 0.0
        terminated = False
        truncated = False
        done = False

        power_list = []
        temp_list = []  # stores Tavg
        t_in_list = []
        t_out_list = []

        ou_state = 0.0  # reset OU noise each episode

        # The env uses max_steps=200; match it here
        for t in range(env.max_steps):
            print(f"Episode {ep}, step {t}:")
            if t == 0:
                reset = True
            else:
                reset = False
            # 1) policy action u in [-1, 1]
            u = agent.take_action(obs)
            # 2) Add exploration noise (applied to u)
            noise = get_noise()
            u_noisy = float(np.clip(u + noise, -1.0, 1.0))

            # 3) u -> physical action: action_rate = reactivity change rate per second
            action_rate = env.denormalize_action(u_noisy)
            print("Action:", action_rate)
            # 4) Update accumulated reactivity: rho_accum += action_rate * interval
            rho_accum = float(rho_accum + action_rate * env.interval)
            print("Accumulated reactivity:", rho_accum)
            # Optional: clamp rho_accum to the allowed range (the env also terminates with -100)
            rho_accum = float(np.clip(rho_accum, env.rho_accum_min, env.rho_accum_max))

            # 5) Build control_variable and write rho_accum into [-3]
            control_variable = []
            control_variable.append(str(rho_accum))
            control_variable.append(str(t))
            control_variable.append(str(float(t+1)))

            # 6) One environment step
            next_obs, reward, terminated, truncated, info_step = env.step(reset, control_variable)

            print("Normalised state:", next_obs, '\n')
            done = bool(terminated) or bool(truncated)
            ep_reward += float(reward)

            if isinstance(info_step, dict):
                if "P" in info_step:
                    power_list.append(float(info_step["P"]))
                if "Tavg" in info_step:
                    temp_list.append(float(info_step["Tavg"]))
                if "t_in" in info_step:
                    t_in_list.append(float(info_step["t_in"]))
                if "t_out" in info_step:
                    t_out_list.append(float(info_step["t_out"]))

            # 7) Store the transition in the replay buffer
            # Note: store action as u_noisy (the normalised action) in the replay buffer
            replay_buffer.add(obs, u_noisy, reward, next_obs, done)

            obs = next_obs

            # 8) Sample and update (dict form required)
            if replay_buffer.size() >= minimal_size and (len(replay_buffer.buffer) if hasattr(replay_buffer, "buffer") else replay_buffer.size()) >= batch_size:
                # Sample
                s, a, r, s2, d = replay_buffer.sample(batch_size)

                transition_dict = {
                    "states": s,
                    "actions": a,
                    "rewards": r,
                    "next_states": s2,
                    "dones": d,
                }
                agent.update(transition_dict)

            if done:
                break

        # Noise decay
        noise_std = max(0.02, noise_std * noise_decay)

        episode_rewards.append(ep_reward)

        # Save power/temperature traces per episode (each file: t_index, value)
        P_ref = float(info["P_ref"])
        T_ref = float(info["T_ref"])

        # ====== Score the episode by real control performance, not by reward ======
        if len(power_list) > 0:
            P_arr = np.array(power_list, dtype=float)
            P_ref = float(info["P_ref"])

            n = len(P_arr)

            # 1. Mean absolute error over the whole episode
            mean_abs_error = np.mean(np.abs(P_arr - P_ref))

            # 2. Maximum undershoot: dropping too far below the target is undesirable
            undershoot = max(0.0, P_ref - np.min(P_arr))

            # 3. Tail mean error: whether the final segment settles near the target
            tail_len = min(100, n)
            tail = P_arr[-tail_len:]
            tail_mae = np.mean(np.abs(tail - P_ref))

            # 4. Settling time: enter the +-2 MW band and hold it until the end
            band = 2.0
            settling_time = n

            for i in range(n):
                if np.all(np.abs(P_arr[i:] - P_ref) <= band):
                    settling_time = i
                    break

            settling_ratio = settling_time / n

            # 5. Overall score; lower is better
            score = (
                    0.5 * mean_abs_error
                    + 2.0 * undershoot
                    + 2.0 * tail_mae
                    + 5.0 * settling_ratio
            )

        else:
            mean_abs_error = np.nan
            undershoot = np.nan
            tail_mae = np.nan
            settling_ratio = np.nan
            score = 1e6

        episode_scores.append(float(score))

        # ====== Save the best model so far ======
        if score < best_score:
            best_score = float(score)
            best_ep = int(ep)

            agent.save_models(save_dir=model_dir, tag="best")
            replay_buffer.save(os.path.join(model_dir, "replay_buffer_best.txt"))

            print(f"Better model found: episode {ep}, best_score={best_score:.3f}")

        # Append to the CSV after each episode
        with open(reward_txt_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                float(ep_reward),
                float(score),
                float(mean_abs_error),
                float(undershoot),
                float(tail_mae),
                float(settling_ratio)
            ])

        print(
            f"[Episode {ep:04d}] "
            f"reward={ep_reward:.3f}, "
            f"score={score:.3f}, "
            f"MAE={mean_abs_error:.3f}, "
            f"undershoot={undershoot:.3f}, "
            f"tail_mae={tail_mae:.3f}, "
            f"settling_ratio={settling_ratio:.3f}, "
            f"steps={t + 1}, done={done}"
        )

        power_path = os.path.join(power_dir, f"power_ep_{ep:04d}_P{P_ref}.csv")
        temp_path = os.path.join(temp_dir, f"temp_ep_{ep:04d}_T{T_ref}.csv")

        with open(power_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_index", "P"])
            for i, v in enumerate(power_list):
                writer.writerow([i, v])

        # temp csv (4 columns)
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_index", "t_in", "t_out", "Tavg"])
            for i in range(len(temp_list)):
                writer.writerow([i, t_in_list[i], t_out_list[i], temp_list[i]])

        # print(f"[Episode {ep:04d}] reward={ep_reward:.3f}, steps={t + 1}, done={done}")


        # Periodic checkpointing
        if ep % save_every == 0:
            tag = f"ep_{ep:04d}"

            # Snapshot the model every 10 episodes without overwriting
            agent.save_models(save_dir=model_dir, tag=tag)

            # Save the latest model (overwrite) so training can resume
            agent.save_models(save_dir=model_dir, tag="latest")

            # Keep only the latest replay buffer (overwrite)
            replay_buffer.save(os.path.join(model_dir, "replay_buffer.txt"))

            print(f"Saved episode {ep} snapshot, latest model and replay buffer to: {model_dir}")

    print(f"\nSaved: {reward_txt_path}")

    if len(episode_scores) >= 5:
        best_scores = sorted(episode_scores)[:5]
        return float(np.mean(best_scores))
    else:
        return float(np.min(episode_scores))


# =========================
# Configure hyperparameters and start training here
# =========================
if __name__ == "__main__":
    # Train only the 776 MW (0.8) condition
    # Without reward_weights the defaults in Env.py are used:
    # wP=1.0, wI=0.015, wD=0.02
    env = QinshanEnv(interval=1.0, target_fractions=(0.8,))

    state_dim = 7
    action_dim = 1
    hidden_dim = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DDPG(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        device=device,
    )

    replay_buffer = ReplayBuffer(capacity=15000)

    train_ddpg(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=200,
        batch_size=64,
        minimal_size=2000,
        noise_type="gaussian",
        noise_std=0.2,
        noise_decay=0.99,
        save_every=10,
        device=device,
        reward_txt_path="./rewards_776_baseline_200.csv",
        power_dir="./power_logs_776_baseline_200",
        temp_dir="./temp_logs_776_baseline_200",
        model_dir="./models_776_baseline",
    )