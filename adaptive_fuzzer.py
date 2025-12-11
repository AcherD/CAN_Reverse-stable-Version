# adaptive_fuzzer.py
import time
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import can

from CAN_fuzzer import CANFuzzer  # 复用你的 CANFuzzer :contentReference[oaicite:4]{index=4}
from VisionDetector import VisionDetector  # 复用你的视觉检测器 :contentReference[oaicite:5]{index=5}


class VisionGuidedAdaptiveFuzzer:
    """
    Vision-guided adaptive CAN fuzzer.

    - 以 CAN ID 为“臂”，使用简单的 epsilon-greedy bandit 做 ID 选择；
    - 对每个 ID 做按字节的 payload 探索（byte-scan）：
        * 初期：依次激活各个 Byte (0..7)，其他字节为 0x00；
        * 完成一轮扫描后：优先重复 reward 较高的字节；
    - 每个 episode 之前/之后通过 VisionDetector 采样仪表盘灯状态，
      根据灯状态变化数量 + 新模式覆盖计算 reward；
    - 当前版本仅实现 byte-level 探索（已经可以显著优于纯随机），
      bit-level 与时序模式探索可以在此基础上继续扩展。
    """

    def __init__(
        self,
        can_fuzzer: CANFuzzer,
        detector: VisionDetector,
        id_start: int,
        id_end: int,
        epsilon: float = 0.2,
        alpha: float = 1.0,
        beta: float = 5.0,
        default_freq_hz: float = 10.0,
        frames_per_episode: int = 20,
        settle_time: float = 0.2,
    ):
        """
        :param can_fuzzer: 已初始化的 CANFuzzer，用于访问 bus 和 ID 范围
        :param detector: 已初始化的 VisionDetector
        :param id_start: 初始 ID 范围起始（含）
        :param id_end: 初始 ID 范围结束（含）
        :param epsilon: ε-greedy 中的探索概率
        :param alpha: reward 中“灯状态变化数量”的权重
        :param beta: reward 中“新模式覆盖”的权重
        :param default_freq_hz: 每个 episode 内发送报文的频率（Hz）
        :param frames_per_episode: 每个 episode 发送的帧数
        :param settle_time: 在发送前后采样灯状态的“稳定等待时间”（秒）
        """
        self.can_fuzzer = can_fuzzer
        self.detector = detector

        # CAN ID 候选集（可以根据需要改成更稀疏/动态扩展）
        self.id_candidates: List[int] = list(range(int(id_start), int(id_end) + 1))

        # bandit 统计：平均 reward 和尝试次数
        self.id_stats: Dict[int, Dict[str, float]] = {
            cid: {"R": 0.0, "N": 0.0} for cid in self.id_candidates
        }

        # per-ID 的 payload 探索状态（目前只做 byte-scan）
        # state = {
        #   "stage": "byte_scan",
        #   "next_byte": 0,
        #   "byte_rewards": [0.0]*8,
        #   "byte_counts": [0]*8
        # }
        self.id_payload_state: Dict[int, Dict] = {
            cid: {
                "stage": "byte_scan",
                "next_byte": 0,
                "byte_rewards": [0.0] * 8,
                "byte_counts": [0] * 8,
            }
            for cid in self.id_candidates
        }

        # Vision 相关
        self.labels: List[str] = detector.labels  # 你的 VisionDetector 中的标签列表 :contentReference[oaicite:6]{index=6}
        self.K: int = len(self.labels)

        # 覆盖集合：记录已经出现过的灯状态模式
        self.coverage = set()

        # 超参数
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.default_freq_hz = float(default_freq_hz)
        self.frames_per_episode = int(frames_per_episode)
        self.settle_time = float(settle_time)

        # 便于停止的标志
        self._stop_flag = False

    # ========= 公共接口 =========

    def stop(self):
        """外部可调用，用于提前结束 run() 循环。"""
        self._stop_flag = True

    def run(self, num_episodes: int = 500):
        """
        执行若干 episode 的自适应 fuzzing。
        每个 episode 选择 (ID, payload)，发送一串报文，并通过视觉反馈更新策略。

        :param num_episodes: 总 episode 数
        """
        print(
            f"[AdaptiveFuzzer] Start vision-guided fuzzing: "
            f"{num_episodes} episodes, ID range=0x{self.id_candidates[0]:X}-0x{self.id_candidates[-1]:X}"
        )

        for ep in range(1, num_episodes + 1):
            if self._stop_flag:
                print(f"[AdaptiveFuzzer] Stopped externally at episode {ep}.")
                break

            # 1. 选择 ID（epsilon-greedy）
            cid = self._select_id()

            # 2. 根据该 ID 对应的 payload 探索状态选择 payload
            payload_bytes, byte_index = self._select_payload_for_id(cid)

            # 3. 在发送前，稍作等待并采样“前状态”
            time.sleep(self.settle_time)
            L_before = self._get_light_state()

            # 4. 执行一次 episode：在给定频率下发送 frames_per_episode 帧
            self._send_episode(cid, payload_bytes)

            # 5. 再次等待并采样“后状态”
            time.sleep(self.settle_time)
            L_after = self._get_light_state()

            # 6. 计算 reward（灯变化 + 新覆盖）
            reward, changed_cnt, is_new = self._compute_reward(L_before, L_after)

            # 7. 更新 bandit 统计（针对该 ID）
            self._update_bandit(cid, reward)

            # 8. 更新 payload 探索统计（针对该 ID 的 Byte）
            self._update_payload_state(cid, byte_index, reward)

            # 9. 打印调试信息
            print(
                f"[EP {ep:04d}] ID=0x{cid:03X}, payload={payload_bytes.hex()}, "
                f"Δlights={changed_cnt}, new={is_new}, reward={reward:.2f}, "
                f"R_id={self.id_stats[cid]['R']:.2f}, N_id={int(self.id_stats[cid]['N'])}"
            )

        print("[AdaptiveFuzzer] Fuzzing finished.")

    # ========= 核心子模块 =========

    def _select_id(self) -> int:
        """
        epsilon-greedy 选择一个 CAN ID：
        - 以概率 ε 选择“探索”：优先选择尝试次数 N 最小的 ID；
        - 以概率 (1-ε) 选择“利用”：选择当前平均 reward R 最大的 ID。
        """
        if random.random() < self.epsilon:
            # 探索：选择尝试次数最少的一批 ID 中的一个
            min_N = min(stat["N"] for stat in self.id_stats.values())
            candidates = [cid for cid, stat in self.id_stats.items() if stat["N"] == min_N]
            cid = random.choice(candidates)
        else:
            # 利用：选择平均 reward 最高的 ID
            max_R = max(stat["R"] for stat in self.id_stats.values())
            candidates = [cid for cid, stat in self.id_stats.items() if stat["R"] == max_R]
            cid = random.choice(candidates)

        return cid

    def _select_payload_for_id(self, cid: int) -> Tuple[bytes, int]:
        """
        针对给定 ID，按“字节扫描（byte-scan）”策略选择 payload。
        当前版本策略：
        1. 若尚未让 0..7 所有字节都至少尝试一次：按顺序逐个字节设置为 0xFF；
        2. 若已扫描过所有字节：优先选择平均 reward 较高的字节。
        返回：
        - payload_bytes: bytes(8)
        - byte_index: 本次激活的字节索引，用于后续更新 byte_rewards。
        """
        state = self.id_payload_state[cid]
        byte_rewards = state["byte_rewards"]
        byte_counts = state["byte_counts"]

        # 判断是否“第一轮扫描尚未完成”
        if min(byte_counts) == 0:
            # 还有没试过的字节：选择第一个 count == 0 的字节
            byte_index = byte_counts.index(0)
        else:
            # 都试过一轮：选择平均 reward 最高的字节
            max_R = max(byte_rewards)
            # 可能有多个相同最大值，从中随机挑一个
            candidates = [
                idx for idx, br in enumerate(byte_rewards) if br == max_R
            ]
            byte_index = random.choice(candidates)

        # 构造 payload：默认全 0，仅将某个字节设为 0xFF
        data = [0x00] * 8
        data[byte_index] = 0xFF
        payload_bytes = bytes(data)

        # 记录下一次要扫描的字节（如果仍在“首轮扫描”过程中）
        if byte_counts[byte_index] == 0:
            state["next_byte"] = (byte_index + 1) % 8

        return payload_bytes, byte_index

    def _send_episode(self, cid: int, payload_bytes: bytes):
        """
        在当前 episode 内，以 default_freq_hz 频率发送 frames_per_episode 帧相同报文。
        """
        interval = 1.0 / self.default_freq_hz
        bus: can.Bus = self.can_fuzzer.bus  # 直接复用 CANFuzzer 的 bus :contentReference[oaicite:7]{index=7}

        msg = can.Message(
            arbitration_id=cid,
            data=payload_bytes,
            is_extended_id=(cid > 0x7FF)
        )

        for _ in range(self.frames_per_episode):
            try:
                bus.send(msg)
            except can.CanError as e:
                print(f"[AdaptiveFuzzer] CAN send failed: {e}")
                # 短暂休眠后继续（或根据需要重试/终止）
            time.sleep(interval)

    def _get_light_state(self) -> Tuple[int, ...]:
        """
        调用 VisionDetector.detect()，根据当前帧的检测结果生成长度为 K 的灯状态向量。
        :return: 例如 (0,1,0,1,...)
        """
        detections = self.detector.detect()
        state = [0] * self.K
        if not detections:
            return tuple(state)

        # 以“至少检测到一次就认为该灯为 1”为准
        label_to_idx = {lbl: idx for idx, lbl in enumerate(self.labels)}
        for det in detections:
            lbl = det["label"]
            if lbl in label_to_idx:
                state[label_to_idx[lbl]] = 1

        return tuple(state)

    def _compute_reward(
        self,
        L_before: Tuple[int, ...],
        L_after: Tuple[int, ...],
    ) -> Tuple[float, int, bool]:
        """
        根据前后灯状态计算 reward：
        r = alpha * (#changed) + beta * (is_new_pattern)
        """
        # 灯状态变化数量
        changed_cnt = sum(1 for b, a in zip(L_before, L_after) if b != a)

        # 新覆盖判定
        is_new = L_after not in self.coverage
        if is_new:
            self.coverage.add(L_after)

        reward = self.alpha * changed_cnt + (self.beta if is_new else 0.0)
        return reward, changed_cnt, is_new

    def _update_bandit(self, cid: int, reward: float):
        """
        使用简单的增量平均更新 R[ID], N[ID]。
        """
        stat = self.id_stats[cid]
        N_old = stat["N"]
        R_old = stat["R"]
        N_new = N_old + 1.0
        # 增量平均：R_new = R_old + (reward - R_old)/N_new
        R_new = R_old + (reward - R_old) / N_new
        stat["N"] = N_new
        stat["R"] = R_new

    def _update_payload_state(self, cid: int, byte_index: int, reward: float):
        """
        更新该 ID 对应字节的 reward 统计（平均 reward）。
        """
        state = self.id_payload_state[cid]
        br = state["byte_rewards"]
        bc = state["byte_counts"]

        N_old = bc[byte_index]
        R_old = br[byte_index]
        N_new = N_old + 1
        R_new = R_old + (reward - R_old) / float(N_new)

        bc[byte_index] = N_new
        br[byte_index] = R_new

        # 如需扩展为 bit-scan / pattern-scan，可在这里根据统计结果切换 stage
        # 例如：
        # if all(c >= MIN_TRIALS for c in bc) and max(br) > SOME_THRESHOLD:
        #     state["stage"] = "bit_scan"
        #     ...
