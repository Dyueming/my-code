import numpy as np
from gymnasium import spaces
import gymnasium as gym
from 修改输入文件_自启动 import Control_self_start
from 读取输出数据 import Data_Extract
import os
import time
import pygetwindow as gw
import re
import subprocess

relap5_path = r'D:\relap5\RELAP5MOD4.0\relap5.exe'
i_path = r'D:\relap5\qinshan-r.i'
steady_output_path = r'D:\relap5\qinshan-r.o'
steady_restart_path = r'D:\relap5\qinshan-r.r'
input_path = r'D:\relap5\RELAP5MOD4.0\qinshan-r.i'
output_path = r'D:\relap5\RELAP5MOD4.0\qinshan-r.o'
restart_path = r'D:\relap5\RELAP5MOD4.0\qinshan-r.r'
file_path = os.path.abspath('.') + r'\实时参数数据存储文件.txt'
extract_path = r'D:\relap5\RELAP5MOD4.0\（relap5）提取数据用'

# 初始化python与relap5交互程序
qinshan_restart = Control_self_start(relap5_path, i_path, steady_output_path, steady_restart_path, input_path, output_path, restart_path, 0)
extract = Data_Extract(output_path, file_path, extract_path)

class QinshanEnv():
    """
    只负责归一化：
    obs = [P_norm, eP_norm, Tavg_norm, eT_norm, rho_accum_norm, dP_norm, dT_norm]
    """

    def __init__(
        self,
        interval=1.0,
        # 动作范围：反应性变化率（每秒）
        action_rate_min=-0.00308,
        action_rate_max=0.00308,
        # 累计反应性范围
        rho_accum_min=-0.107,
        rho_accum_max=0.107,
        # 变化率归一化上限（你给的先验）
        max_abs_dP=10.0,    # MW/s
        max_abs_dT=0.5,     # K/s
        # 只训练这四个目标（100->90/80/70/60；训练目标可在reset里切）
        target_fractions=(0.9, 0.8, 0.7, 0.6),
        reward_weights=None,

    ):
        super().__init__()
        self.interval = float(interval)
        self.max_steps = 200  # 你自己定
        self.step_count = 0

        if reward_weights is None:
            reward_weights = {
                "wP": 1.0,
                "wI": 0.015,
                "wD": 0.02,
            }

        self.reward_weights = reward_weights

        self.action_rate_min = float(action_rate_min)
        self.action_rate_max = float(action_rate_max)

        self.rho_accum_min = float(rho_accum_min)
        self.rho_accum_max = float(rho_accum_max)

        self.max_abs_dP = float(max_abs_dP)
        self.max_abs_dT = float(max_abs_dT)

        # 动作空间：网络输出到[-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 观测空间：固定[-1,1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # 目标点（写死：你给的数据）
        self.ref_table = {
            1.0: (970.0, 571.78),
            0.9: (873.0, 569.165),
            0.8: (776.0, 566.44),
            0.7: (679.0, 563.66),
            0.6: (582.0, 560.84),
        }

        self.target_fractions = tuple(float(x) for x in target_fractions)

        # 归一化用范围：用 60~100 覆盖区间
        self.P_min = self.ref_table[0.6][0]
        self.P_max = self.ref_table[1.0][0]
        self.P_span = self.P_max - self.P_min

        self.T_min = self.ref_table[0.6][1]
        self.T_max = self.ref_table[1.0][1]
        self.T_span = self.T_max - self.T_min

        # rho归一化中心/尺度
        self.rho_center = 0.5 * (self.rho_accum_min + self.rho_accum_max)
        self.rho_scale = 0.5 * (self.rho_accum_max - self.rho_accum_min)

        # 内部变量（用于算导数）
        self.last_P = None
        self.last_Tavg = None

        self.prev_E_err = None
        self.int_E_err = 0.0
        self.max_abs_int_E_err = 125.0  # 积分上限：需要按实际效果调
        self.max_abs_dE_err = 50.0  # 微分裁剪上限：需要按实际效果调

    @staticmethod
    def _clip(x, lo=-1.0, hi=1.0):
        return float(np.clip(x, lo, hi))

    def set_target(self, target_frac: float):
        target_frac = float(target_frac)
        if target_frac not in self.ref_table:
            raise ValueError(f"target_frac must be in {list(self.ref_table.keys())}")
        self.target_frac = target_frac
        self.P_ref, self.T_ref = self.ref_table[target_frac]

    # ---------------- 动作归一化/反归一化 ----------------
    def normalize_action(self, action_rate: float) -> float:
        """(可选) 物理动作 -> 网络动作[-1,1]"""
        u = (action_rate - self.action_rate_min) / (self.action_rate_max - self.action_rate_min + 1e-12)  # [0,1]
        u = 2*u - 1
        return self._clip(u)

    def denormalize_action(self, u: float) -> float:
        """网络动作[-1,1] -> 物理反应性变化率"""
        u = float(u)
        a = 0.5 * (u + 1.0) * (self.action_rate_max - self.action_rate_min) + self.action_rate_min
        # 再裁一下避免数值误差
        return float(np.clip(a, self.action_rate_min, self.action_rate_max))

    # ---------------- 归一化 obs ----------------
    def build_obs(self, P: float, Tavg: float, rho_accum: float):
        """
        你在自己的 step() 里拿到仿真新状态后调用：
        obs = env.build_obs(P, Tavg, rho_accum)
        """
        P = float(P)
        Tavg = float(Tavg)
        rho_accum = float(rho_accum)

        # 当前误差
        eP = P - self.P_ref
        eT = Tavg - self.T_ref

        # 归一化到[-1,1]
        # 1) P_norm：用min-max线性映射到[-1,1]
        P_norm = self._clip(2.0 * (P - self.P_min) / (self.P_span + 1e-12) - 1.0)

        # 2) eP_norm：按“半跨度”缩放，让误差在目标范围内大致落在[-1,1]
        eP_norm = self._clip(eP / ( self.P_max-self.P_ref + 1e-12))

        # 3) Tavg_norm：min-max映射
        T_norm = self._clip(2.0 * (Tavg - self.T_min) / (self.T_span + 1e-12) - 1.0)

        # 4) eT_norm：按半跨度缩放
        eT_norm = self._clip(eT / ( self.T_max-self.T_ref + 1e-12))

        # 5) rho_accum_norm
        rho_accum_norm = self._clip((rho_accum - self.rho_center) / (self.rho_scale + 1e-12))

        # 6) dP_norm, dT_norm：差分/你给的上限
        if self.last_P is None:
            dP = 0.0
        else:
            dP = (P - self.last_P) / self.interval

        if self.last_Tavg is None:
            dT = 0.0
        else:
            dT = (Tavg - self.last_Tavg) / self.interval

        dP_norm = self._clip(dP / (self.max_abs_dP + 1e-12))
        dT_norm = self._clip(dT / (self.max_abs_dT + 1e-12))

        # 更新 last，用于下一步导数
        self.last_P = P
        self.last_Tavg = Tavg

        obs = np.array([P_norm, eP_norm, T_norm, eT_norm, rho_accum_norm, dP_norm, dT_norm], dtype=np.float32)
        return obs

    # ---------------- reset：只做初始化 ----------------
    def reset(self,options=None):

        # 目标切换：外部可传 target_frac，不传则随机从训练目标集合选
        if options is not None and "target_frac" in options:
            self.set_target(options["target_frac"])
        else:
            self.set_target(np.random.choice(self.target_fractions))

        # 初始从100%状态开始（你前面说通常从1.0阶跃到目标）
        self.P = self.ref_table[1.0][0]
        self.Tavg = self.ref_table[1.0][1]
        self.rho_accum = 0.0  # TODO：如果你初始累计反应性不是0，请改这里

        # 初始化 last 以便第一步 dP/dT=0（或你也可以让它未定义）
        self.last_P = None
        self.last_Tavg = None

        self.step_count = 0

        obs = self.build_obs(self.P, self.Tavg, self.rho_accum)
        info = {"target_frac": self.target_frac, "P_ref": self.P_ref, "T_ref": self.T_ref}

        self.prev_E_err = None
        self.int_E_err = 0.0
        self.prev_E_err = None
        return obs, info

    # ---------------- step：留给你自己写 ----------------
    def step(self, reset, control_variable):
        self.step_count += 1
        qinshan_restart.modify_input_file(reset, control_variable)
        time.sleep(0.1)
        qinshan_restart.self_starting()
        time.sleep(0.1)
        done_flag = extract.extract_data()

        window_title = 'Visual Fortran run-time error'
        windowt = gw.getWindowsWithTitle(window_title)
        if windowt:
            # 获取第一个窗口
            windowm = windowt[0]
            # 关闭窗口
            windowm.close()

            program_name = "relap5.exe"  # 使用tasklist命令获取当前运行的进程列表
            process_list = subprocess.run(["tasklist"], capture_output=True, text=True).stdout  # 检查程序是否存在
            if program_name in process_list:  # 程序存在，使用taskkill命令关闭程序
                subprocess.run(["taskkill", "/F", "/IM", program_name])
                print(f"程序 {program_name} 已关闭。")
            else:
                print(f"程序 {program_name} 未找到。")
        else:
            pass
        try:
            windows = gw.getWindowsWithTitle('relap5')
            while len(windows) != 0:
                # 使用pygetwindow库关闭窗口
                window = windows[0]
                window.close()
                # 等待窗口关闭
                windows = gw.getWindowsWithTitle('relap5')

        except:
            pass

        # 提取datatotal文件数据
        try:
            with open('D:\\relap5\\RELAP5MOD4.0\\（relap5）提取数据用\\datatotal', 'r', encoding='latin-1') as f1:
                lines = f1.readlines()
                content_list = lines[-1].rstrip('\n\r') if lines else ''
                float_array = [float(x) for x in content_list.split()]
            time.sleep(0.1)

            P_k = float_array[-1] / (10 ** 6)
            t_in = float_array[-3]
            t_out = float_array[-2]
        except:
            f = subprocess.Popen('start look.exe', cwd=extract_path, shell=True)  # 启动look.exe文件提取数据
            f.wait()
            time.sleep(0.2)  # 等待datatotal文件更新后读取信息
            with open('D:\\relap5\\RELAP5MOD4.0\\（relap5）提取数据用\\datatotal', 'r', encoding='latin-1') as f1:
                lines = f1.readlines()
                content_list = lines[-1].rstrip('\n\r') if lines else ''
                float_array = [float(x) for x in content_list.split()]
            time.sleep(0.1)

            P_k = float_array[-1] / (10 ** 6)
            t_in = float_array[-3]
            t_out = float_array[-2]
        self.P = float(P_k)
        self.Tavg = float(0.5 * (t_in + t_out))
        self.rho_accum = float(control_variable[-3])
        print("功率：",self.P,"温度：",t_in,t_out,"平均温度：",self.Tavg)

        obs = self.build_obs(self.P, self.Tavg, self.rho_accum)

        # ========= 奖励：P + I + D（用 E_err 代价）=========
        eP = float(obs[1])  # eP_norm
        eT = float(obs[3])  # eT_norm

        # -------- 奖励权重：由外部优化器传入 --------
        wP = float(self.reward_weights.get("wP", 1.0))
        wI = float(self.reward_weights.get("wI", 0.015))
        wD = float(self.reward_weights.get("wD", 0.02))

        # -------- P项（比例误差惩罚）--------
        E_err = abs(eP)
        dt = float(self.interval)
        # -------- I项（积分误差惩罚）--------
        self.int_E_err += E_err * dt
        self.int_E_err = float(np.clip(self.int_E_err, -self.max_abs_int_E_err, self.max_abs_int_E_err))

        # -------- D项（微分误差惩罚）--------
        if self.prev_E_err is None:
            dE_err = 0.0
        else:
            dE_err = (E_err - self.prev_E_err) / (dt + 1e-12)
        dE_err = float(np.clip(dE_err, -self.max_abs_dE_err, self.max_abs_dE_err))

        # 为了避免“刚开始训练/刚 reset 后”抖动时 D 过强：
        # 你 env 的 self.step_count 从0开始计步，所以可以做一个随步数爬坡
        warmup_steps = 20
        wD_eff = wD * min(1.0, self.step_count / warmup_steps)

        # 组合成奖励：代价越大越差 => 奖励取负号
        reward = -(wP * E_err + wI * self.int_E_err + wD_eff * dE_err)
        # reward = -(wP * E_err + wI * self.int_E_err + wD_eff * abs(dE_err))

        # 更新上一时刻
        self.prev_E_err = E_err

        # 物理超限惩罚（建议只在“超限”时覆盖/加惩罚，不要把 done_flag 正常结束也覆盖）
        rho_out_of_limit = (
                (self.rho_accum < self.rho_accum_min) or (self.rho_accum > self.rho_accum_max)
        )

        terminated = bool(done_flag) or rho_out_of_limit

        if rho_out_of_limit:
            reward = -10.0
        print(f'比例奖励：{E_err}，积分奖励：{self.int_E_err}，微分奖励：{dE_err}',f'总奖励：{reward}')

        # truncated 由步数决定（如果你暂时不想管时间上限，就先设 False）
        truncated = (self.step_count >= self.max_steps)

        info = {
            "target_frac": self.target_frac,
            "P_ref": float(self.P_ref),
            "T_ref": float(self.T_ref),
            "P": float(self.P),
            "Tavg": float(self.Tavg),
            "eP_norm": float(eP),
            "eT_norm": float(eT),
            "t_in": float(t_in),
            "t_out": float(t_out),
        }

        return obs, reward, terminated, truncated, info