# adaptive_fuzzer.py
import time
import random
import os
import json
import csv
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import can

from CAN_fuzzer import CANFuzzer
from VisionDetector import VisionDetector


class VisionGuidedAdaptiveFuzzer:
    """
    Vision-guided adaptive CAN fuzzer (byte + bit exploration + neighbor expansion).

    功能概述：
    - 以 CAN ID 为“臂”，使用 epsilon-greedy multi-armed bandit 进行 ID 选择；
    - 针对每个 ID，执行分阶段 payload 探索：
        * 阶段1：按字节扫描（byte_scan），识别“活跃字节”；
        * 阶段2：按 bit 扫描（bit_scan），在活跃字节内逐 bit 试探；
    - 针对平均 reward 较高的 ID 做邻域扩展（neighborhood expansion）：
        * 自动将 0x(ID±Δ) 加入候选 ID 集；
    - 每个 episode 利用 VisionDetector 在发送前后采样仪表盘灯状态，
      根据灯状态变化 + 新灯态覆盖计算 reward；
    - 运行结束后，自动生成：
        * CSV 日志：逐 episode 的详细记录（ID、payload、灯态、reward 等）；
        * 文本报告：按 ID 汇总平均 reward、覆盖模式数以及高价值 payload 模式；
        * JSON 报告（方便后续程序化分析）。
    """

    def __init__(
        self,
        can_fuzzer: CANFuzzer,
        detector: VisionDetector,
        id_start: int,
        id_end: int,
        seed_ids: Optional[List[int]] = None,
        epsilon: float = 0.2,
        alpha: float = 1.0,
        beta: float = 5.0,
        default_freq_hz: float = 10.0,
        frames_per_episode: int = 20,
        settle_time: float = 0.2,
        # bit-scan 相关阈值
        min_byte_trials_for_bit: int = 2,
        byte_reward_threshold_for_bit: float = 1.0,
        # 邻域扩展相关参数
        neighbor_delta: int = 0x1,
        neighbor_reward_threshold: float = 2.0,
        neighbor_min_trials: int = 5,
        # 日志相关
        log_dir: str = "logs",
    ):
        """
        :param can_fuzzer: 已初始化的 CANFuzzer，用于访问 bus 和 ID 范围
        :param detector: 已初始化的 VisionDetector
        :param id_start: 全局 ID 范围起始（含）
        :param id_end: 全局 ID 范围结束（含）
        :param seed_ids: 可选，仅以这些 ID 作为初始候选集；若为 None，则用 [id_start, id_end]
        :param epsilon: ε-greedy 中的探索概率
        :param alpha: reward 中“灯状态变化数量”的权重
        :param beta: reward 中“新模式覆盖”的权重
        :param default_freq_hz: 每个 episode 内发送报文的频率（Hz）
        :param frames_per_episode: 每个 episode 发送的帧数
        :param settle_time: 在发送前后采样灯状态的“稳定等待时间”（秒）
        :param min_byte_trials_for_bit: 进入 bit-scan 前，对每个字节至少尝试次数
        :param byte_reward_threshold_for_bit: 进入 bit-scan 的“活跃字节”平均 reward 下限
        :param neighbor_delta: 邻域扩展时的步长（例如 0x1、0x10）
        :param neighbor_reward_threshold: 触发邻域扩展的最小平均 reward
        :param neighbor_min_trials: 触发邻域扩展的最小 episode 次数
        :param log_dir: 日志与报告输出目录
        """
        self.can_fuzzer = can_fuzzer
        self.detector = detector

        self.id_min = int(id_start)
        self.id_max = int(id_end)

        # 初始 ID 候选集
        if seed_ids is not None and len(seed_ids) > 0:
            self.id_candidates: List[int] = sorted(
                cid for cid in set(int(x) for x in seed_ids)
                if self.id_min <= cid <= self.id_max
            )
        else:
            self.id_candidates = list(range(self.id_min, self.id_max + 1))

        # bandit 统计：平均 reward 和尝试次数
        self.id_stats: Dict[int, Dict[str, float]] = {
            cid: {"R": 0.0, "N": 0.0} for cid in self.id_candidates
        }

        # per-ID 的 payload 探索状态（byte + bit）
        # state = {
        #   "stage": "byte_scan" / "bit_scan",
        #   "byte_rewards": [0.0]*8,
        #   "byte_counts": [0]*8,
        #   "next_byte": 0,
        #   "neighbor_expanded": False,
        #   "target_byte": None,
        #   "bit_rewards": [0.0]*8 or None,
        #   "bit_counts": [0]*8 or None,
        #   "next_bit": 0 or None,
        # }
        self.id_payload_state: Dict[int, Dict] = {}
        for cid in self.id_candidates:
            self.id_payload_state[cid] = self._init_payload_state()

        # Vision 相关
        self.labels: List[str] = detector.labels
        self.K: int = len(self.labels)

        # 全局覆盖集合：记录已经出现过的灯状态模式
        self.coverage = set()

        # 每个 ID 对应的灯态模式集合（便于 summary）
        self.id_patterns: Dict[int, set] = defaultdict(set)

        # 逐 episode 日志
        self.episode_log: List[Dict] = []

        # 每个 ID + payload 的统计（count, total_reward, max_reward）
        self.id_payload_stats: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)

        # 超参数
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.default_freq_hz = float(default_freq_hz)
        self.frames_per_episode = int(frames_per_episode)
        self.settle_time = float(settle_time)
        self.min_byte_trials_for_bit = int(min_byte_trials_for_bit)
        self.byte_reward_threshold_for_bit = float(byte_reward_threshold_for_bit)
        self.neighbor_delta = int(neighbor_delta)
        self.neighbor_reward_threshold = float(neighbor_reward_threshold)
        self.neighbor_min_trials = int(neighbor_min_trials)

        # 日志目录与 run_id
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        结束时自动输出日志和报告。
        """
        print(
            f"[AdaptiveFuzzer] Start vision-guided fuzzing: "
            f"{num_episodes} episodes, ID global range=0x{self.id_min:X}-0x{self.id_max:X}, "
            f"initial IDs={len(self.id_candidates)}"
        )

        try:
            for ep in range(1, num_episodes + 1):
                if self._stop_flag:
                    print(f"[AdaptiveFuzzer] Stopped externally at episode {ep}.")
                    break

                # 1. 选择 ID（epsilon-greedy）
                cid = self._select_id()

                # 2. 根据该 ID 对应的 payload 探索状态选择 payload
                payload_bytes, byte_index, bit_index = self._select_payload_for_id(cid)

                # 3. 发送前等待，采样前状态
                time.sleep(self.settle_time)
                L_before = self._get_light_state()

                # 4. 执行一次 episode：在给定频率下发送 frames_per_episode 帧
                self._send_episode(cid, payload_bytes)

                # 5. 再次等待并采样后状态
                time.sleep(self.settle_time)
                L_after = self._get_light_state()

                # 6. 计算 reward（灯变化 + 新覆盖）
                reward, changed_cnt, is_new = self._compute_reward(L_before, L_after)

                # 7. 更新 bandit 统计（针对该 ID）
                self._update_bandit(cid, reward)

                # 8. 更新 payload 探索状态（byte / bit）
                self._update_payload_state(cid, byte_index, bit_index, reward)

                # 9. 邻域扩展（在 ID 平均 reward 较高、尝试次数够多时）
                self._maybe_expand_neighbors(cid)

                # 10. 记录 ID 专属灯态覆盖
                self.id_patterns[cid].add(L_after)

                # 11. 记录 episode log
                self._log_episode(
                    ep=ep,
                    cid=cid,
                    payload_bytes=payload_bytes,
                    stage=self.id_payload_state[cid]["stage"],
                    byte_index=byte_index,
                    bit_index=bit_index,
                    reward=reward,
                    changed_cnt=changed_cnt,
                    is_new=is_new,
                    L_before=L_before,
                    L_after=L_after,
                )

                # 12. 打印调试信息
                print(
                    f"[EP {ep:04d}] ID=0x{cid:03X}, payload={payload_bytes.hex()}, "
                    f"stage={self.id_payload_state[cid]['stage']}, "
                    f"byte={byte_index}, bit={bit_index if bit_index is not None else '-'}, "
                    f"Δlights={changed_cnt}, new={is_new}, reward={reward:.2f}, "
                    f"R_id={self.id_stats[cid]['R']:.2f}, N_id={int(self.id_stats[cid]['N'])}"
                )

        finally:
            # 不论是否中断，都尝试保存日志与报告
            self._save_logs_and_report()
            print("[AdaptiveFuzzer] Fuzzing finished and report generated.")

    # ========= 内部辅助：payload 状态初始化 =========

    def _init_payload_state(self) -> Dict:
        return {
            "stage": "byte_scan",
            "byte_rewards": [0.0] * 8,
            "byte_counts": [0] * 8,
            "next_byte": 0,
            "neighbor_expanded": False,
            "target_byte": None,
            "bit_rewards": None,
            "bit_counts": None,
            "next_bit": None,
        }

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

    def _select_payload_for_id(self, cid: int) -> Tuple[bytes, int, Optional[int]]:
        """
        针对给定 ID，根据该 ID 的探索阶段选择 payload。
        - byte_scan 阶段：按字节扫描；
        - bit_scan 阶段：在选定 target_byte 内按 bit 扫描。
        返回：
        - payload_bytes: bytes(8)
        - byte_index: 激活的字节索引
        - bit_index: 若在 bit_scan，则为激活的 bit 索引；否则为 None。
        """
        state = self.id_payload_state[cid]
        stage = state["stage"]

        if stage == "byte_scan":
            payload_bytes, byte_index = self._select_payload_byte_scan(state)
            bit_index = None
        elif stage == "bit_scan":
            payload_bytes, byte_index, bit_index = self._select_payload_bit_scan(state)
        else:
            # 理论上不会到达；为安全起见，fallback 到 byte_scan
            payload_bytes, byte_index = self._select_payload_byte_scan(state)
            bit_index = None

        return payload_bytes, byte_index, bit_index

    def _select_payload_byte_scan(self, state: Dict) -> Tuple[bytes, int]:
        """
        BYTE_SCAN 策略：
        1. 若还有未尝试的字节（byte_counts == 0），按顺序逐个扫描；
        2. 否则，选择平均 reward 最高的字节重复试探。
        payload：默认全 0，仅将该字节置为 0xFF。
        """
        byte_rewards = state["byte_rewards"]
        byte_counts = state["byte_counts"]

        # 第一轮扫描：优先选择尚未尝试过的字节
        if min(byte_counts) == 0:
            byte_index = byte_counts.index(0)
        else:
            # 所有字节都试过：选择平均 reward 最高的字节
            max_R = max(byte_rewards)
            candidates = [idx for idx, br in enumerate(byte_rewards) if br == max_R]
            byte_index = random.choice(candidates)

        data = [0x00] * 8
        data[byte_index] = 0xFF
        payload_bytes = bytes(data)

        # 更新“下一字节”指针（仅用于信息记录，可选）
        state["next_byte"] = (byte_index + 1) % 8

        return payload_bytes, byte_index

    def _select_payload_bit_scan(self, state: Dict) -> Tuple[bytes, int, int]:
        """
        BIT_SCAN 策略：
        - 在 target_byte 内扫描 bit：
            * 若还有 bit_counts==0 的位，按顺序扫描；
            * 否则选择平均 reward 最高的 bit。
        payload：仅将 target_byte 的某一位设为 1，其余全 0。
        """
        target_byte = state["target_byte"]
        bit_rewards = state["bit_rewards"]
        bit_counts = state["bit_counts"]

        if bit_rewards is None or bit_counts is None or target_byte is None:
            # 防御：若状态不完整则退回到 byte_scan（极少发生）
            payload_bytes, byte_index = self._select_payload_byte_scan(state)
            return payload_bytes, byte_index, -1

        # 若还有未尝试的 bit
        if min(bit_counts) == 0:
            bit_index = bit_counts.index(0)
        else:
            max_R = max(bit_rewards)
            candidates = [idx for idx, br in enumerate(bit_rewards) if br == max_R]
            bit_index = random.choice(candidates)

        data = [0x00] * 8
        data[target_byte] = 1 << bit_index
        payload_bytes = bytes(data)

        # 更新下一 bit 指针（可选）
        state["next_bit"] = (bit_index + 1) % 8

        return payload_bytes, target_byte, bit_index

    def _send_episode(self, cid: int, payload_bytes: bytes):
        """
        在当前 episode 内，以 default_freq_hz 频率发送 frames_per_episode 帧相同报文。
        """
        interval = 1.0 / self.default_freq_hz
        bus: can.Bus = self.can_fuzzer.bus

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
        changed_cnt = sum(1 for b, a in zip(L_before, L_after) if b != a)

        is_new = L_after not in self.coverage
        if is_new:
            self.coverage.add(L_after)

        reward = self.alpha * changed_cnt + (self.beta if is_new else 0.0)
        return reward, changed_cnt, is_new

    def _update_bandit(self, cid: int, reward: float):
        """
        使用增量平均更新 R[ID], N[ID]。
        """
        stat = self.id_stats[cid]
        N_old = stat["N"]
        R_old = stat["R"]
        N_new = N_old + 1.0
        R_new = R_old + (reward - R_old) / N_new
        stat["N"] = N_new
        stat["R"] = R_new

    def _update_payload_state(
        self,
        cid: int,
        byte_index: int,
        bit_index: Optional[int],
        reward: float,
    ):
        """
        更新该 ID 对应的 payload 探索统计：
        - 在 byte_scan 阶段：更新 byte_rewards/byte_counts；
        - 在 bit_scan 阶段：更新 bit_rewards/bit_counts；
        并在 byte_scan 阶段满足条件时升级到 bit_scan。
        """
        state = self.id_payload_state[cid]
        stage = state["stage"]

        # 1) BYTE 级统计
        byte_rewards = state["byte_rewards"]
        byte_counts = state["byte_counts"]
        if 0 <= byte_index < 8:
            N_old = byte_counts[byte_index]
            R_old = byte_rewards[byte_index]
            N_new = N_old + 1
            R_new = R_old + (reward - R_old) / float(N_new)
            byte_counts[byte_index] = N_new
            byte_rewards[byte_index] = R_new

        # 2) 若当前已经是 bit_scan，则更新 bit 级统计
        if stage == "bit_scan" and bit_index is not None and bit_index >= 0:
            bit_rewards = state["bit_rewards"]
            bit_counts = state["bit_counts"]
            if bit_rewards is not None and bit_counts is not None and 0 <= bit_index < 8:
                N_old = bit_counts[bit_index]
                R_old = bit_rewards[bit_index]
                N_new = N_old + 1
                R_new = R_old + (reward - R_old) / float(N_new)
                bit_counts[bit_index] = N_new
                bit_rewards[bit_index] = R_new
            return

        # 3) 若仍处于 byte_scan 阶段，检查是否应该升级为 bit_scan
        if stage == "byte_scan":
            # 条件1：所有字节至少尝试 min_byte_trials_for_bit 次
            if min(byte_counts) >= self.min_byte_trials_for_bit:
                # 条件2：存在平均 reward 足够高的“活跃字节”
                max_R = max(byte_rewards)
                if max_R >= self.byte_reward_threshold_for_bit:
                    active_bytes = [
                        idx for idx, br in enumerate(byte_rewards) if br == max_R
                    ]
                    target_byte = random.choice(active_bytes)

                    state["stage"] = "bit_scan"
                    state["target_byte"] = target_byte
                    state["bit_rewards"] = [0.0] * 8
                    state["bit_counts"] = [0] * 8
                    state["next_bit"] = 0

    def _maybe_expand_neighbors(self, cid: int):
        """
        若某个 ID 的平均 reward 较高且尝试次数足够，则在其邻域添加新的 ID。
        邻域定义为 [cid - neighbor_delta, cid + neighbor_delta] 交 [id_min, id_max]。
        每个 ID 只扩展一次。
        """
        state = self.id_payload_state[cid]
        if state.get("neighbor_expanded", False):
            return

        stat = self.id_stats[cid]
        if stat["N"] < self.neighbor_min_trials:
            return
        if stat["R"] < self.neighbor_reward_threshold:
            return

        new_ids = []
        for delta in (-self.neighbor_delta, self.neighbor_delta):
            nid = cid + delta
            if nid < self.id_min or nid > self.id_max:
                continue
            if nid in self.id_stats:
                continue
            new_ids.append(nid)

        if not new_ids:
            # 没有可扩展的邻域 ID
            state["neighbor_expanded"] = True
            return

        for nid in new_ids:
            self.id_stats[nid] = {"R": 0.0, "N": 0.0}
            self.id_payload_state[nid] = self._init_payload_state()
            self.id_candidates.append(nid)
            print(f"[AdaptiveFuzzer] Neighbor ID added: 0x{nid:03X}")

        state["neighbor_expanded"] = True

    # ========= 日志 & 报告 =========

    def _log_episode(
        self,
        ep: int,
        cid: int,
        payload_bytes: bytes,
        stage: str,
        byte_index: int,
        bit_index: Optional[int],
        reward: float,
        changed_cnt: int,
        is_new: bool,
        L_before: Tuple[int, ...],
        L_after: Tuple[int, ...],
    ):
        """
        将本次 episode 的信息追加到内存日志，并更新“高价值 ID + payload 模式”统计。
        """
        rec = {
            "episode": ep,
            "id": cid,
            "id_hex": f"0x{cid:03X}",
            "payload_hex": payload_bytes.hex(),
            "stage": stage,
            "byte_index": byte_index,
            "bit_index": bit_index if bit_index is not None else -1,
            "reward": float(reward),
            "changed_cnt": int(changed_cnt),
            "is_new_pattern": bool(is_new),
            "L_before": "".join(str(x) for x in L_before),
            "L_after": "".join(str(x) for x in L_after),
        }
        self.episode_log.append(rec)

        # 更新 ID + payload 模式的聚合统计
        per_id = self.id_payload_stats[cid]
        key = payload_bytes.hex()
        if key not in per_id:
            per_id[key] = {
                "count": 0,
                "total_reward": 0.0,
                "max_reward": 0.0,
            }
        entry = per_id[key]
        entry["count"] += 1
        entry["total_reward"] += float(reward)
        entry["max_reward"] = max(entry["max_reward"], float(reward))
        per_id[key] = entry
        self.id_payload_stats[cid] = per_id

    def _save_logs_and_report(self):
        """
        将 episode 级别日志写入 CSV，并生成一份汇总报告（txt + json）。
        """
        if not self.episode_log:
            print("[AdaptiveFuzzer] No episodes recorded, skip logging.")
            return

        base_name = f"adaptive_fuzz_{self.run_id}"
        csv_path = os.path.join(self.log_dir, base_name + "_episodes.csv")
        txt_path = os.path.join(self.log_dir, base_name + "_summary.txt")
        json_path = os.path.join(self.log_dir, base_name + "_summary.json")

        # 1) 写 CSV 日志
        fieldnames = list(self.episode_log[0].keys())
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self.episode_log:
                    writer.writerow(rec)
            print(f"[AdaptiveFuzzer] Episode log written to {csv_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write CSV log: {e}")

        # 2) 生成汇总统计（内存中的 summary_dict）
        summary_dict = self._build_summary_dict()

        # 2.1 写 JSON（便于后处理）
        try:
            with open(json_path, "w", encoding="utf-8") as f_json:
                json.dump(summary_dict, f_json, indent=2, ensure_ascii=False)
            print(f"[AdaptiveFuzzer] Summary JSON written to {json_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write JSON summary: {e}")

        # 2.2 写文本报告，便于人工阅读
        try:
            with open(txt_path, "w", encoding="utf-8") as f_txt:
                self._write_human_readable_report(summary_dict, f_txt)
            print(f"[AdaptiveFuzzer] Summary report written to {txt_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write text summary: {e}")

    def _build_summary_dict(self) -> Dict:
        """
        汇总得到一个结构化的 summary_dict，用于 JSON 和文本报告。
        """
        total_episodes = len(self.episode_log)
        total_patterns = len(self.coverage)

        # 只统计 N>0 的 ID
        id_summary = []
        for cid, stat in self.id_stats.items():
            if stat["N"] <= 0:
                continue
            per_id_payload = self.id_payload_stats.get(cid, {})
            patterns = self.id_patterns.get(cid, set())
            id_summary.append({
                "id": cid,
                "id_hex": f"0x{cid:03X}",
                "episodes": int(stat["N"]),
                "mean_reward": float(stat["R"]),
                "payload_variants": len(per_id_payload),
                "pattern_count": len(patterns),
            })

        # 按 mean_reward 排序
        id_summary_sorted = sorted(
            id_summary, key=lambda x: x["mean_reward"], reverse=True
        )

        # 为每个 ID 选出若干高价值 payload 模式
        per_id_payload_top = {}
        for cid, payload_stats in self.id_payload_stats.items():
            # 跳过完全没试过的 ID
            if cid not in [x["id"] for x in id_summary_sorted]:
                continue
            items = []
            for payload_hex, s in payload_stats.items():
                count = s["count"]
                total_reward = s["total_reward"]
                max_reward = s["max_reward"]
                avg_reward = total_reward / count if count > 0 else 0.0
                items.append({
                    "payload_hex": payload_hex,
                    "count": count,
                    "avg_reward": avg_reward,
                    "max_reward": max_reward,
                })
            # 按 avg_reward 排序，取前若干个
            items_sorted = sorted(
                items, key=lambda x: x["avg_reward"], reverse=True
            )
            per_id_payload_top[str(cid)] = items_sorted[:5]

        summary_dict = {
            "run_id": self.run_id,
            "global_id_range": [self.id_min, self.id_max],
            "episodes": total_episodes,
            "unique_patterns": total_patterns,
            "label_mapping": {i: lbl for i, lbl in enumerate(self.labels)},
            "id_summary": id_summary_sorted,
            "id_payload_top": per_id_payload_top,
        }
        return summary_dict

    def _write_human_readable_report(self, summary: Dict, f_txt):
        """
        将 summary_dict 以结构化文本形式写入文件，方便你直接阅读。
        """
        f_txt.write(f"Vision-Guided Adaptive CAN Fuzzing Report\n")
        f_txt.write(f"Run ID: {summary['run_id']}\n")
        f_txt.write(
            f"Global ID range: 0x{summary['global_id_range'][0]:03X} "
            f"- 0x{summary['global_id_range'][1]:03X}\n"
        )
        f_txt.write(f"Total episodes: {summary['episodes']}\n")
        f_txt.write(f"Total unique light patterns: {summary['unique_patterns']}\n\n")

        f_txt.write("Warning-light index mapping:\n")
        for idx, lbl in summary["label_mapping"].items():
            f_txt.write(f"  [{idx}] {lbl}\n")
        f_txt.write("\n")

        # Top IDs
        f_txt.write("Top IDs by mean reward (descending):\n")
        for entry in summary["id_summary"]:
            f_txt.write(
                f"- ID {entry['id_hex']}: episodes={entry['episodes']}, "
                f"mean_reward={entry['mean_reward']:.2f}, "
                f"payload_variants={entry['payload_variants']}, "
                f"pattern_count={entry['pattern_count']}\n"
            )
        f_txt.write("\n")

        # 对每个 ID 列出高价值 payload 模式
        f_txt.write("High-value payload patterns per ID:\n\n")
        for entry in summary["id_summary"]:
            cid = entry["id"]
            id_hex = entry["id_hex"]
            f_txt.write(f"ID {id_hex}:\n")
            payload_list = summary["id_payload_top"].get(str(cid), [])
            if not payload_list:
                f_txt.write("  (no payload statistics)\n\n")
                continue
            for p in payload_list:
                f_txt.write(
                    f"  payload={p['payload_hex']}, "
                    f"count={p['count']}, "
                    f"avg_reward={p['avg_reward']:.2f}, "
                    f"max_reward={p['max_reward']:.2f}\n"
                )
            f_txt.write("\n")
