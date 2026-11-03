import numpy as np
from gymnasium import spaces
import gymnasium as gym
from relap5_input import Control_self_start
from relap5_output import Data_Extract
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
file_path = os.path.abspath('.') + r'\realtime_params.txt'
extract_path = r'D:\relap5\RELAP5MOD4.0\relap5-extract'

# Initialise the Python <-> RELAP5 bridge
qinshan_restart = Control_self_start(relap5_path, i_path, steady_output_path, steady_restart_path, input_path, output_path, restart_path, 0)
extract = Data_Extract(output_path, file_path, extract_path)

class QinshanEnv():
    """
    Handles normalisation only:
    obs = [P_norm, eP_norm, Tavg_norm, eT_norm, rho_accum_norm, dP_norm, dT_norm]
    """

    def __init__(
        self,
        interval=1.0,
        # Action range: reactivity change rate per second
        action_rate_min=-0.00308,
        action_rate_max=0.00308,
        # Accumulated reactivity range
        rho_accum_min=-0.107,
        rho_accum_max=0.107,
        # Normalisation cap for the rate (prior value given)
        max_abs_dP=10.0,    # MW/s
        max_abs_dT=0.5,     # K/s
        # Train only these targets (100 -> 90/80/70/60; switchable in reset)
        target_fractions=(0.9, 0.8, 0.7, 0.6),
        reward_weights=None,

    ):
        super().__init__()
        self.interval = float(interval)
        self.max_steps = 200  # set as needed
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

        # Action space: network output in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: fixed to [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Target table (hard-coded reference data)
        self.ref_table = {
            1.0: (970.0, 571.78),
            0.9: (873.0, 569.165),
            0.8: (776.0, 566.44),
            0.7: (679.0, 563.66),
            0.6: (582.0, 560.84),
        }

        self.target_fractions = tuple(float(x) for x in target_fractions)

        # Normalisation range: 60-100 covers the operating band
        self.P_min = self.ref_table[0.6][0]
        self.P_max = self.ref_table[1.0][0]
        self.P_span = self.P_max - self.P_min

        self.T_min = self.ref_table[0.6][1]
        self.T_max = self.ref_table[1.0][1]
        self.T_span = self.T_max - self.T_min

        # rho normalisation centre / scale
        self.rho_center = 0.5 * (self.rho_accum_min + self.rho_accum_max)
        self.rho_scale = 0.5 * (self.rho_accum_max - self.rho_accum_min)

        # Internal state (for derivatives)
        self.last_P = None
        self.last_Tavg = None

        self.prev_E_err = None
        self.int_E_err = 0.0
        self.max_abs_int_E_err = 125.0  # integral limit; tune per run
        self.max_abs_dE_err = 50.0  # derivative clipping limit; tune per run

    @staticmethod
    def _clip(x, lo=-1.0, hi=1.0):
        return float(np.clip(x, lo, hi))

    def set_target(self, target_frac: float):
        target_frac = float(target_frac)
        if target_frac not in self.ref_table:
            raise ValueError(f"target_frac must be in {list(self.ref_table.keys())}")
        self.target_frac = target_frac
        self.P_ref, self.T_ref = self.ref_table[target_frac]

    # ---------------- action normalisation / denormalisation ----------------
    def normalize_action(self, action_rate: float) -> float:
        """(Optional) Physical action -> network action in [-1, 1]"""
        u = (action_rate - self.action_rate_min) / (self.action_rate_max - self.action_rate_min + 1e-12)  # [0,1]
        u = 2*u - 1
        return self._clip(u)

    def denormalize_action(self, u: float) -> float:
        """Network action in [-1, 1] -> physical reactivity change rate"""
        u = float(u)
        a = 0.5 * (u + 1.0) * (self.action_rate_max - self.action_rate_min) + self.action_rate_min
        # Clip again to avoid numerical drift
        return float(np.clip(a, self.action_rate_min, self.action_rate_max))

    # ---------------- normalise obs ----------------
    def build_obs(self, P: float, Tavg: float, rho_accum: float):
        """
        Call this from your own step() after obtaining the new simulated state:
        obs = env.build_obs(P, Tavg, rho_accum)
        """
        P = float(P)
        Tavg = float(Tavg)
        rho_accum = float(rho_accum)

        # Current error
        eP = P - self.P_ref
        eT = Tavg - self.T_ref

        # Normalise to [-1, 1]
        # 1) P_norm: min-max linear mapping to [-1, 1]
        P_norm = self._clip(2.0 * (P - self.P_min) / (self.P_span + 1e-12) - 1.0)

        # 2) eP_norm: scale by half-span so the error stays roughly in [-1, 1]
        eP_norm = self._clip(eP / ( self.P_max-self.P_ref + 1e-12))

        # 3) Tavg_norm: min-max mapping
        T_norm = self._clip(2.0 * (Tavg - self.T_min) / (self.T_span + 1e-12) - 1.0)

        # 4) eT_norm: scale by half-span
        eT_norm = self._clip(eT / ( self.T_max-self.T_ref + 1e-12))

        # 5) rho_accum_norm
        rho_accum_norm = self._clip((rho_accum - self.rho_center) / (self.rho_scale + 1e-12))

        # 6) dP_norm, dT_norm: difference / given upper bounds
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

        # Update last for the next derivative
        self.last_P = P
        self.last_Tavg = Tavg

        obs = np.array([P_norm, eP_norm, T_norm, eT_norm, rho_accum_norm, dP_norm, dT_norm], dtype=np.float32)
        return obs

    # ---------------- reset: initialisation only ----------------
    def reset(self,options=None):

        # Target selection: pass target_frac, otherwise sample from the training set
        if options is not None and "target_frac" in options:
            self.set_target(options["target_frac"])
        else:
            self.set_target(np.random.choice(self.target_fractions))

        # Start from 100% power (typically a step from 1.0 to the target)
        self.P = self.ref_table[1.0][0]
        self.Tavg = self.ref_table[1.0][1]
        self.rho_accum = 0.0  # TODO: change this if the initial accumulated reactivity is not 0

        # Initialise last so that dP/dT = 0 on the first step
        self.last_P = None
        self.last_Tavg = None

        self.step_count = 0

        obs = self.build_obs(self.P, self.Tavg, self.rho_accum)
        info = {"target_frac": self.target_frac, "P_ref": self.P_ref, "T_ref": self.T_ref}

        self.prev_E_err = None
        self.int_E_err = 0.0
        self.prev_E_err = None
        return obs, info

    # ---------------- step: left for you to implement ----------------
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
            # Grab the first window
            windowm = windowt[0]
            # Close the window
            windowm.close()

            program_name = "relap5.exe"  # running process list via tasklist
            process_list = subprocess.run(["tasklist"], capture_output=True, text=True).stdout  # check whether the process exists
            if program_name in process_list:  # found the process, close it with taskkill
                subprocess.run(["taskkill", "/F", "/IM", program_name])
                print(f"Process {program_name} closed.")
            else:
                print(f"Process {program_name} not found.")
        else:
            pass
        try:
            windows = gw.getWindowsWithTitle('relap5')
            while len(windows) != 0:
                # Close the window via pygetwindow
                window = windows[0]
                window.close()
                # Wait for the window to close
                windows = gw.getWindowsWithTitle('relap5')

        except:
            pass

        # Parse the datatotal file
        try:
            with open('D:\\relap5\\RELAP5MOD4.0\\relap5-extract\\datatotal', 'r', encoding='latin-1') as f1:
                lines = f1.readlines()
                content_list = lines[-1].rstrip('\n\r') if lines else ''
                float_array = [float(x) for x in content_list.split()]
            time.sleep(0.1)

            P_k = float_array[-1] / (10 ** 6)
            t_in = float_array[-3]
            t_out = float_array[-2]
        except:
            f = subprocess.Popen('start look.exe', cwd=extract_path, shell=True)  # launch look.exe to extract data
            f.wait()
            time.sleep(0.2)  # wait for the datatotal file before reading
            with open('D:\\relap5\\RELAP5MOD4.0\\relap5-extract\\datatotal', 'r', encoding='latin-1') as f1:
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
        print("Power:", self.P, "Temp:", t_in, t_out, "Avg temp:", self.Tavg)

        obs = self.build_obs(self.P, self.Tavg, self.rho_accum)

        # ========= Reward: P + I + D (using the E_err cost) =========
        eP = float(obs[1])  # eP_norm
        eT = float(obs[3])  # eT_norm

        # -------- Reward weights: supplied by the outer optimiser --------
        wP = float(self.reward_weights.get("wP", 1.0))
        wI = float(self.reward_weights.get("wI", 0.015))
        wD = float(self.reward_weights.get("wD", 0.02))

        # -------- P term (proportional error penalty) --------
        E_err = abs(eP)
        dt = float(self.interval)
        # -------- I term (integral error penalty) --------
        self.int_E_err += E_err * dt
        self.int_E_err = float(np.clip(self.int_E_err, -self.max_abs_int_E_err, self.max_abs_int_E_err))

        # -------- D term (derivative error penalty) --------
        if self.prev_E_err is None:
            dE_err = 0.0
        else:
            dE_err = (E_err - self.prev_E_err) / (dt + 1e-12)
        dE_err = float(np.clip(dE_err, -self.max_abs_dE_err, self.max_abs_dE_err))

        # To avoid an overly strong D term right after training starts or a reset:
        # step_count starts at 0, so the D weight can ramp up with the step count
        warmup_steps = 20
        wD_eff = wD * min(1.0, self.step_count / warmup_steps)

        # Combine into a reward: a larger cost is worse, so negate
        reward = -(wP * E_err + wI * self.int_E_err + wD_eff * dE_err)
        # reward = -(wP * E_err + wI * self.int_E_err + wD_eff * abs(dE_err))

        # Update the previous step
        self.prev_E_err = E_err

        # Physical-limit penalty (only override on a violation, not on normal termination)
        rho_out_of_limit = (
                (self.rho_accum < self.rho_accum_min) or (self.rho_accum > self.rho_accum_max)
        )

        terminated = bool(done_flag) or rho_out_of_limit

        if rho_out_of_limit:
            reward = -10.0
        print(f'P term: {E_err}, I term: {self.int_E_err}, D term: {dE_err}', f'Total reward: {reward}')

        # truncated is driven by the step count (set False to ignore the time cap)
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