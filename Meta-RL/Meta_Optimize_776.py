import os
import json
import optuna
import torch

from Env import QinshanEnv
from DDPG import ReplayBuffer, DDPG
from Train import train_ddpg


def objective(trial):
    # =========================
    # 1. Optuna 搜索奖励权重
    # =========================
    wP = trial.suggest_float("wP", 0.6, 1.6)
    wI = trial.suggest_float("wI", 0.005, 0.03)
    wD = trial.suggest_float("wD", 0.005, 0.03)

    reward_weights = {
        "wP": float(wP),
        "wI": float(wI),
        "wD": float(wD),
    }

    trial_id = trial.number

    # 每个 trial 独立文件夹，避免互相覆盖
    trial_dir = os.path.join("./meta_logs_776", f"trial_{trial_id:04d}")
    model_dir = os.path.join(trial_dir, "models")
    power_dir = os.path.join(trial_dir, "power_logs")
    temp_dir = os.path.join(trial_dir, "temp_logs")
    reward_path = os.path.join(trial_dir, "rewards.csv")

    os.makedirs(trial_dir, exist_ok=True)

    with open(os.path.join(trial_dir, "reward_weights.json"), "w", encoding="utf-8") as f:
        json.dump(reward_weights, f, indent=4, ensure_ascii=False)

    # =========================
    # 2. 创建 776 MW 单工况环境
    # =========================
    env = QinshanEnv(
        interval=1.0,
        target_fractions=(0.8,),
        reward_weights=reward_weights,
    )

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

    score = train_ddpg(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=100,
        batch_size=64,
        minimal_size=2000,
        noise_type="gaussian",
        noise_std=0.2,
        noise_decay=0.981,
        save_every=10,
        device=device,
        reward_txt_path=reward_path,
        power_dir=power_dir,
        temp_dir=temp_dir,
        model_dir=model_dir,
    )

    print(
        f"\nTrial {trial_id} finished: "
        f"wP={wP:.5f}, wI={wI:.5f}, wD={wD:.5f}, score={score:.5f}"
    )

    return score


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=2026)

    study = optuna.create_study(
        study_name="meta_optimize_776_reward_weights",
        direction="minimize",
        sampler=sampler,
        storage="sqlite:///meta_optimize_776.db",
        load_if_exists=True,
    )

    # =========================
    # 补足 5 个成功完成的 trial
    # =========================
    target_complete_trials = 5

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    failed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.FAIL
    ]

    running_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.RUNNING
    ]

    remaining_trials = max(0, target_complete_trials - len(completed_trials))

    print("\n========== 当前 Optuna 进度 ==========")
    print(f"已成功完成 trial 数：{len(completed_trials)}")
    print(f"失败 trial 数：{len(failed_trials)}")
    print(f"运行中 trial 数：{len(running_trials)}")
    print(f"还需要继续运行 trial 数：{remaining_trials}")

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            catch=(PermissionError,)
        )
    else:
        print("已经完成足够数量的 trial，不再继续搜索。")


    print("\n========== 最优结果 ==========")
    print("best_params:", study.best_params)
    print("best_score:", study.best_value)

    os.makedirs("./meta_logs_776", exist_ok=True)

    with open("./meta_logs_776/best_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": study.best_params,
                "best_score": study.best_value,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )