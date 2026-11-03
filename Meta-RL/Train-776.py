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

    # OU 噪声状态（如果你选 ou）
    ou_state = 0.0
    episode_rewards = []
    episode_scores = []

    best_score = float("inf")
    best_ep = None

    os.makedirs(model_dir, exist_ok=True)


    target_fracs = list(env.target_fractions)  # 例如 [0.9, 0.8, 0.7, 0.6]
    counts = {tf: 0 for tf in target_fracs}

    # 初始化 reward/score 记录文件，只写一次表头
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

        # 每回合“随机”，但优先从当前训练次数最少的工况里随机选
        min_count = min(counts[tf] for tf in target_fracs)
        candidates = [tf for tf in target_fracs if counts[tf] == min_count]
        chosen_tf = float(np.random.choice(candidates))

        obs, info = env.reset(options={"target_frac": chosen_tf})
        counts[chosen_tf] += 1

        print("目标功率：", info["P_ref"])

        rho_accum = float(env.rho_accum)  # 初始通常是 0.0（来自你的 env.reset）

        ep_reward = 0.0
        terminated = False
        truncated = False
        done = False

        power_list = []
        temp_list = []  # 存 Tavg
        t_in_list = []
        t_out_list = []

        ou_state = 0.0  # 每回合重置OU噪声

        # 你 env 内部 max_steps=200，这里也用它更一致
        for t in range(env.max_steps):
            print(f"第{ep}回合，第{t}步：")
            if t == 0:
                reset = True
            else:
                reset = False
            # 1) 策略动作 u ∈ [-1,1]
            u = agent.take_action(obs)
            # 2) 加探索噪声（作用在 u 上）
            noise = get_noise()
            u_noisy = float(np.clip(u + noise, -1.0, 1.0))

            # 3) u -> 物理动作：action_rate = 反应性变化率（每秒）
            action_rate = env.denormalize_action(u_noisy)
            print("动作：",action_rate)
            # 4) 更新累计反应性：rho_accum += action_rate * interval
            rho_accum = float(rho_accum + action_rate * env.interval)
            print("累计反应性：",rho_accum)
            # 可选：把 rho_accum 限制在环境允许范围内（环境里也会终止并给 -100）
            rho_accum = float(np.clip(rho_accum, env.rho_accum_min, env.rho_accum_max))

            # 5) 构造 control_variable，写入 rho_accum 到 [-3]
            control_variable = []
            control_variable.append(str(rho_accum))
            control_variable.append(str(t))
            control_variable.append(str(float(t+1)))

            # 6) 环境一步
            next_obs, reward, terminated, truncated, info_step = env.step(reset, control_variable)

            print("归一化状态：",next_obs, '\n')
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

            # 7) 写入经验回放
            # 注意：replay_buffer 里 action 建议存 u_noisy（归一化动作）
            replay_buffer.add(obs, u_noisy, reward, next_obs, done)

            obs = next_obs

            # 8) 采样更新（需要 dict 形式）
            if replay_buffer.size() >= minimal_size and (len(replay_buffer.buffer) if hasattr(replay_buffer, "buffer") else replay_buffer.size()) >= batch_size:
                # 采样
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

        # 噪声衰减
        noise_std = max(0.02, noise_std * noise_decay)

        episode_rewards.append(ep_reward)

        # 保存每回合的功率/温度轨迹（每个文件：两列：t_index, value）
        P_ref = float(info["P_ref"])
        T_ref = float(info["T_ref"])

        # ====== 用真实控制效果评价本回合，不用 reward 评价 ======
        if len(power_list) > 0:
            P_arr = np.array(power_list, dtype=float)
            P_ref = float(info["P_ref"])

            n = len(P_arr)

            # 1. 全过程平均绝对误差
            mean_abs_error = np.mean(np.abs(P_arr - P_ref))

            # 2. 最大下冲量：降功率任务中，低于目标太多是不理想的
            undershoot = max(0.0, P_ref - np.min(P_arr))

            # 3. 末段平均误差：看最后一段是否稳定接近目标
            tail_len = min(100, n)
            tail = P_arr[-tail_len:]
            tail_mae = np.mean(np.abs(tail - P_ref))

            # 4. 稳定时间：进入 ±2 MW 误差带并保持到结束
            band = 2.0
            settling_time = n

            for i in range(n):
                if np.all(np.abs(P_arr[i:] - P_ref) <= band):
                    settling_time = i
                    break

            settling_ratio = settling_time / n

            # 5. 综合评价值，越小越好
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

        # ====== 保存历史最佳模型 ======
        if score < best_score:
            best_score = float(score)
            best_ep = int(ep)

            agent.save_models(save_dir=model_dir, tag="best")
            replay_buffer.save(os.path.join(model_dir, "replay_buffer_best.txt"))

            print(f"发现更优模型：Episode {ep}, best_score={best_score:.3f}")

        # 每个 episode 结束后追加写入 CSV
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

        # temp csv（4列）
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_index", "t_in", "t_out", "Tavg"])
            for i in range(len(temp_list)):
                writer.writerow([i, t_in_list[i], t_out_list[i], temp_list[i]])

        # print(f"[Episode {ep:04d}] reward={ep_reward:.3f}, steps={t + 1}, done={done}")


        # 定期保存
        if ep % save_every == 0:
            tag = f"ep_{ep:04d}"

            # 每10回合保存一版模型，不覆盖
            agent.save_models(save_dir=model_dir, tag=tag)

            # 保存 latest 模型，覆盖更新，方便后续继续训练
            agent.save_models(save_dir=model_dir, tag="latest")

            # 经验池只保存最新版本，直接覆盖
            replay_buffer.save(os.path.join(model_dir, "replay_buffer.txt"))

            print(f"已保存 Episode {ep} 模型快照、latest模型和最新经验池到：{model_dir}")

    print(f"\n已保存：{reward_txt_path}")

    if len(episode_scores) >= 5:
        best_scores = sorted(episode_scores)[:5]
        return float(np.mean(best_scores))
    else:
        return float(np.min(episode_scores))


# =========================
# 你需要在这里配置超参数并启动训练
# =========================
if __name__ == "__main__":
    # 只训练 776 MW 工况，也就是 0.8 工况
    # 不传 reward_weights，使用 Env.py 里的默认权重：
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