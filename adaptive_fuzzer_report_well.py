import os
import time
import random
import json
import csv
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import can

from CAN_fuzzer import CANFuzzer
from VisionDetector import VisionDetector
import os
import time
import random
import json
import csv
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import can

from CAN_fuzzer import CANFuzzer
from VisionDetector import VisionDetector

from collections import defaultdict

class SuperVisionGuidedAdaptiveFuzzer:
    """
    Vision-guided adaptive CAN fuzzer.

    关键特点：
    - ID 选择：全局覆盖 + epsilon-greedy bandit（带单 ID 最大尝试数）
    - 单 ID 试验：baseline payload vs mutated payload 成对对比
    - payload：结构化随机（全随机、稀疏激活、围绕高价值 payload 微调）
    - 时间窗口：mutated payload 默认连续发送多帧（有利于触发“3 次/2s”之类的条件）
    - bit→灯 映射：基于 baseline→mutated 过程中“变动位”和“灯变化”的统计
    - 多 ID 组合：周期性组合多个已知“有趣事件”来尝试触发 Master Warning 等多 ID 故障灯
    - 实验结束后输出：episodes.csv + summary.json + summary.txt + dbc_candidates.pdf
    """

    def __init__(
        self,
        can_fuzzer: CANFuzzer,
        detector: VisionDetector,
        id_start: int,
        id_end: int,
        seed_ids: Optional[List[int]] = None,
        # bandit & reward
        epsilon: float = 0.2,
        alpha: float = 1.0,
        beta: float = 5.0,
        # CAN 发送与时序
        default_freq_hz: float = 20.0,   # 建议稍微提一点频率
        baseline_repeats: int = 0,       # 后面会不再使用 baseline 帧
        mutated_repeats: int = 3,
        settle_time: float = 0.05,       # 现在当作 detection_delay 使用
        lamp_reset_time: float = 0.6,    # 新增：灯自动熄灭时间 + 安全裕量
        # ID 覆盖约束
        global_min_trials_per_id: int = 5,
        max_trials_per_id: int = 500,
        # 邻域扩展
        neighbor_delta: int = 1,
        neighbor_reward_threshold: float = 1.0,
        neighbor_min_trials: int = 10,
        # 多 ID 组合试验
        multi_combo_period: int = 50,
        min_events_for_combo: int = 3,
        # 视觉 warmup
        vision_warmup_time: float = 2.0,
        # bit→灯 映射阈值
        min_bit_events_for_mapping: int = 5,
        min_confidence_for_mapping: float = 0.6,
        # 日志目录
        log_dir: str = "logs",
    ):
        self.can_fuzzer = can_fuzzer
        self.detector = detector

        self.settle_time = float(settle_time)  # 现在把它当作 detection_delay
        self.lamp_reset_time = float(lamp_reset_time) # 两组报文之间的最小等待

        # ID 范围 & 初始候选集
        self.id_min = int(id_start)
        self.id_max = int(id_end)

        if seed_ids:
            self.id_candidates: List[int] = sorted(
                cid for cid in set(int(x) for x in seed_ids)
                if self.id_min <= cid <= self.id_max
            )
        else:
            self.id_candidates = list(range(self.id_min, self.id_max + 1))

        # 每个 ID 的 bandit 统计
        self.id_stats: Dict[int, Dict[str, float]] = {
            cid: {"R": 0.0, "N": 0.0} for cid in self.id_candidates
        }

        # 视觉 labels（YOLO 的类名）
        self.labels: List[str] = getattr(detector, "labels", [])
        self.K: int = len(self.labels)

        # baseline payload（全 0）
        self.baseline_payload: bytes = bytes([0x00] * 8)

        # 覆盖与日志
        self.coverage = set()  # set of tuples(light_state)
        self.id_patterns: Dict[int, set] = defaultdict(set)

        self.trial_log: List[Dict] = []  # 每个“试验”一条记录（single or multi）
        self.interesting_events: List[Dict] = []  # 单 ID 中发现有灯变化的“高价值事件”
        self.id_best_payloads: Dict[int, List[bytes]] = defaultdict(list)

        # multi-ID 组合统计（仅用于报告）
        self.multi_combo_events: List[Dict] = []

        # bandit & reward 参数
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # 发送与时间参数
        self.default_freq_hz = float(default_freq_hz)
        self.baseline_repeats = int(baseline_repeats)
        self.mutated_repeats = int(mutated_repeats)
        self.settle_time = float(settle_time)

        # 覆盖与 bandit 限制
        self.global_min_trials_per_id = int(global_min_trials_per_id)
        self.max_trials_per_id = int(max_trials_per_id)

        # 邻域扩展
        self.neighbor_delta = int(neighbor_delta)
        self.neighbor_reward_threshold = float(neighbor_reward_threshold)
        self.neighbor_min_trials = int(neighbor_min_trials)
        self.neighbor_expanded_ids = set()

        # 多 ID 组合参数
        self.multi_combo_period = int(multi_combo_period)
        self.min_events_for_combo = int(min_events_for_combo)

        # 视觉 warmup
        self.vision_warmup_time = float(vision_warmup_time)

        # bit 映射阈值
        self.min_bit_events_for_mapping = int(min_bit_events_for_mapping)
        self.min_confidence_for_mapping = float(min_confidence_for_mapping)

        # 日志路径
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._stop_flag = False

    # ===================== 公共接口 =====================

    def stop(self):
        """允许外部提前终止 run 循环。"""
        self._stop_flag = True

    def _wait_lamps_reset(self):
        """
        等待仪表盘上的灯自然熄灭。
        固定 sleep lamp_reset_time。
        也可以改成“循环检测直到所有灯都为 0 或超时”。
        """
        time.sleep(self.lamp_reset_time)


    def run(self, num_episodes: int = 5000):
        """
        主循环：执行 num_episodes 个“试验 episode”。
        每个 episode 要么是单 ID baseline vs mutated 试验，要么是多 ID 组合试验。
        """
        print(
            f"[AdaptiveFuzzer] Start fuzzing: {num_episodes} episodes, "
            f"ID range 0x{self.id_min:X}-0x{self.id_max:X}, "
            f"initial IDs={len(self.id_candidates)}"
        )

        self._warmup_vision()

        try:
            for ep in range(1, num_episodes + 1):
                if self._stop_flag:
                    print(f"[AdaptiveFuzzer] Stopped externally at episode {ep}.")
                    break

                # 周期性进行 multi-ID 组合尝试，用于触发 Master Warning 等
                if (
                    self.multi_combo_period > 0
                    and ep % self.multi_combo_period == 0
                    and len(self.interesting_events) >= self.min_events_for_combo
                ):
                    self._run_multi_id_combo_trial(ep)
                    continue

                # 常规：单 ID baseline vs mutated 试验
                self._run_single_id_trial(ep)

        finally:
            self._save_logs_and_report()
            print("[AdaptiveFuzzer] Fuzzing finished, logs & report generated.")

    # ===================== 视觉 warmup =====================

    def _warmup_vision(self):
        """在第一次发送 CAN 报文前预热摄像头和模型。"""
        if self.vision_warmup_time <= 0:
            return
        print(f"[AdaptiveFuzzer] Warming up vision for {self.vision_warmup_time:.1f} s...")
        end_t = time.time() + self.vision_warmup_time
        while time.time() < end_t:
            try:
                _ = self._get_light_state()
            except Exception as e:
                print(f"[AdaptiveFuzzer] Vision warmup error: {e}")
                break
            time.sleep(0.1)

    # ===================== 单 ID 试验（核心） =====================

    def _run_single_id_trial(self, ep: int):
        cid = self._select_id()

        # 1) 等待灯灭 + baseline 视觉
        self._wait_lamps_reset()
        L0 = self._get_light_state()  # 此时应该是全 0 或稳定状态

        # 2) mutated 报文：直接向该 ID 注入 payload
        M = self._generate_mutated_payload(cid)
        self._send_frames(cid, M, repeats=self.mutated_repeats)

        # 短暂等待，让仪表有时间刷新，但保证在 0.5s 内
        time.sleep(self.settle_time)
        L1 = self._get_light_state()

        # 3) reward & 灯变化
        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(L0, L1)

        # 4) 更新 bandit 和邻域
        self._update_bandit(cid, reward)
        self.id_patterns[cid].add(L1)
        self._maybe_expand_neighbors(cid)

        # 5) 记录有趣事件（用于 bit 映射与 multi-ID 组合）
        self._register_interesting_event(cid, M, reward, lamp_on, lamp_off)

        # 6) 日志记录
        rec = {
            "episode": ep,
            "type": "single",
            "id": cid,
            "id_hex": f"0x{cid:03X}",
            "baseline_payload": self.baseline_payload.hex(),  # 概念上的全 0 baseline
            "mut_payload": M.hex(),
            "reward": float(reward),
            "changed_cnt": int(changed_cnt),
            "is_new_pattern": bool(is_new),
            "L_before": "".join(str(x) for x in L0),
            "L_after": "".join(str(x) for x in L1),
            "lamp_on": lamp_on,
            "lamp_off": lamp_off,
        }
        self.trial_log.append(rec)

        print(
            f"[EP {ep:04d}] single ID=0x{cid:03X}, "
            f"mut={M.hex()}, Δlights={changed_cnt}, new={is_new}, "
            f"R_id={self.id_stats[cid]['R']:.2f}, N_id={int(self.id_stats[cid]['N'])}"
        )

    # ===================== 多 ID 组合试验（Master Warning 等） =====================

    def _run_multi_id_combo_trial(self, ep: int):
        if len(self.interesting_events) < self.min_events_for_combo:
            return

        events = random.sample(
            self.interesting_events,
            k=min(3, len(self.interesting_events))
        )

        # 1) 确保起点灯状态干净
        self._wait_lamps_reset()
        L_init = self._get_light_state()

        # 2) 快速依次发送多个 ID 的 payload
        for ev in events:
            cid = ev["id"]
            payload = ev["payload"]
            # 多 ID 组合时，为了不拉长时间，这里只发 1 帧
            self._send_frames(cid, payload, repeats=1, freq_hz=50.0)
            time.sleep(0.01)

        # 3) 短暂等待，保证在 0.5s 内检测
        time.sleep(self.settle_time)
        L_end = self._get_light_state()

        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(L_init, L_end)

        master_idx = self._find_master_warning_index()
        master_on = False
        if master_idx is not None and len(L_init) == len(L_end):
            if L_init[master_idx] == 0 and L_end[master_idx] == 1:
                master_on = True

        rec = {
            "episode": ep,
            "type": "multi",
            "id": -1,
            "id_hex": "multi",
            "sequence_ids": [ev["id"] for ev in events],
            "sequence_payloads": [ev["payload"].hex() for ev in events],
            "reward": float(reward),
            "changed_cnt": int(changed_cnt),
            "is_new_pattern": bool(is_new),
            "master_on": master_on,
            "L_before": "".join(str(x) for x in L_init),
            "L_after": "".join(str(x) for x in L_end),
            "lamp_on": lamp_on,
            "lamp_off": lamp_off,
        }
        self.trial_log.append(rec)
        self.multi_combo_events.append(rec)

        print(
            f"[EP {ep:04d}] multi-ID combo, "
            f"reward={reward:.2f}, Δlights={changed_cnt}, master_on={master_on}"
        )

    # ===================== ID 选择（bandit + 覆盖） =====================

    def _select_id(self) -> int:
        """
        两阶段：
        1) 覆盖：保证所有 ID 至少试验 global_min_trials_per_id 次；
        2) 在此基础上，对未达到 max_trials_per_id 的 ID 用 epsilon-greedy。
        """
        # 覆盖阶段
        min_N = min(stat["N"] for stat in self.id_stats.values())
        if min_N < self.global_min_trials_per_id:
            candidates = [cid for cid, stat in self.id_stats.items() if stat["N"] == min_N]
            return random.choice(candidates)

        # epsilon-greedy 阶段，限制单 ID 最大次数
        eligible_ids = [cid for cid, stat in self.id_stats.items() if stat["N"] < self.max_trials_per_id]
        if not eligible_ids:
            eligible_ids = list(self.id_stats.keys())

        if random.random() < self.epsilon:
            # 探索：在 eligible 中选 N 最小的
            min_N_elig = min(self.id_stats[cid]["N"] for cid in eligible_ids)
            candidates = [cid for cid in eligible_ids if self.id_stats[cid]["N"] == min_N_elig]
        else:
            # 利用：在 eligible 中选 R 最大的
            max_R = max(self.id_stats[cid]["R"] for cid in eligible_ids)
            candidates = [cid for cid in eligible_ids if self.id_stats[cid]["R"] == max_R]

        return random.choice(candidates)

    def _maybe_expand_neighbors(self, cid: int):
        """
        当某个 ID 的平均 reward 较高且尝试数达到一定阈值时，
        将其邻接 ID（cid±neighbor_delta）加入候选集。
        """
        if cid in self.neighbor_expanded_ids:
            return

        stat = self.id_stats[cid]
        if stat["N"] < self.neighbor_min_trials:
            return
        if stat["R"] < self.neighbor_reward_threshold:
            return

        new_ids = []
        for d in (-self.neighbor_delta, self.neighbor_delta):
            nid = cid + d
            if nid < self.id_min or nid > self.id_max:
                continue
            if nid in self.id_stats:
                continue
            new_ids.append(nid)

        if not new_ids:
            self.neighbor_expanded_ids.add(cid)
            return

        for nid in new_ids:
            self.id_stats[nid] = {"R": 0.0, "N": 0.0}
            self.id_candidates.append(nid)
            print(f"[AdaptiveFuzzer] Neighbor ID added: 0x{nid:03X}")

        self.neighbor_expanded_ids.add(cid)

    # ===================== CAN 发送 & 视觉采样 =====================

    def _send_frames(self, cid: int, payload: bytes, repeats: int = 1, freq_hz: Optional[float] = None):
        """
        在当前 ID 上重复发送若干帧 payload。
        freq_hz 为 None 时使用 default_freq_hz。
        """
        if freq_hz is None:
            freq_hz = self.default_freq_hz

        interval = 1.0 / float(freq_hz)
        bus: can.Bus = self.can_fuzzer.bus

        msg = can.Message(
            arbitration_id=cid,
            data=payload,
            is_extended_id=(cid > 0x7FF),
        )

        for _ in range(repeats):
            try:
                bus.send(msg)
            except can.CanError as e:
                print(f"[AdaptiveFuzzer] CAN send failed: {e}")
            time.sleep(interval)

    def _get_light_state(self) -> Tuple[int, ...]:
        """
        从视觉模块获取当前故障灯状态，映射为固定长度 0/1 向量。
        """
        detections = self.detector.detect()
        if not detections:
            return tuple([0] * self.K)

        if not self.labels:
            # 第一次检测时建立 labels 列表
            for det in detections:
                lbl = det["label"]
                if lbl not in self.labels:
                    self.labels.append(lbl)
            self.K = len(self.labels)

        label_to_idx = {lbl: idx for idx, lbl in enumerate(self.labels)}
        state = [0] * self.K

        for det in detections:
            lbl = det["label"]
            if lbl not in label_to_idx:
                # 出现新的 label，动态扩展
                self.labels.append(lbl)
                label_to_idx[lbl] = len(self.labels) - 1
                state.append(1)
                self.K = len(self.labels)
            else:
                state[label_to_idx[lbl]] = 1

        return tuple(state)

    def _find_master_warning_index(self) -> Optional[int]:
        """
        尝试在 labels 中找到 Master warning 灯的索引。
        规则：label 中包含 'Master' 或 'master'。
        """
        for idx, lbl in enumerate(self.labels):
            if "master" in lbl.lower():
                return idx
        return None

    # ===================== reward & event 统计 =====================

    def _compute_reward(
        self,
        L_before: Tuple[int, ...],
        L_after: Tuple[int, ...],
    ) -> Tuple[float, int, bool, List[int], List[int]]:
        """
        reward = alpha * (# 灯变化数) + beta * 1(出现新的灯态模式)
        同时返回哪些灯 0→1 / 1→0。
        """
        if len(L_before) != len(L_after):
            return 0.0, 0, False, [], []

        lamp_on = []
        lamp_off = []
        changed_cnt = 0

        for i, (b, a) in enumerate(zip(L_before, L_after)):
            if b != a:
                changed_cnt += 1
                if b == 0 and a == 1:
                    lamp_on.append(i)
                elif b == 1 and a == 0:
                    lamp_off.append(i)

        is_new = L_after not in self.coverage
        if is_new:
            self.coverage.add(L_after)

        reward = self.alpha * changed_cnt + (self.beta if is_new else 0.0)
        return reward, changed_cnt, is_new, lamp_on, lamp_off

    def _update_bandit(self, cid: int, reward: float):
        """增量更新该 ID 的平均 reward 和尝试次数。"""
        stat = self.id_stats[cid]
        N_old = stat["N"]
        R_old = stat["R"]
        N_new = N_old + 1.0
        R_new = R_old + (reward - R_old) / N_new
        stat["N"] = N_new
        stat["R"] = R_new

    def _register_interesting_event(
        self,
        cid: int,
        payload: bytes,
        reward: float,
        lamp_on: List[int],
        lamp_off: List[int],
        min_reward: float = 1.0,
        max_events_per_id: int = 20,
    ):
        """
        记录“有趣事件”：只要 reward 足够且发生了灯变化，用于：
        - multi-ID 组合实验
        - 后续（如需要）引导型 payload 生成
        """
        if reward < min_reward or not lamp_on:
            return

        labels = [self.labels[i] for i in lamp_on] if self.labels else []

        ev = {
            "id": cid,
            "payload": payload,
            "lamp_on": lamp_on,
            "lamp_off": lamp_off,
            "labels": labels,
            "reward": reward,
        }
        self.interesting_events.append(ev)
        if len(self.interesting_events) > 500:
            self.interesting_events = self.interesting_events[-500:]

        # 针对该 ID 保存若干高价值 payload
        best_list = self.id_best_payloads[cid]
        if payload not in best_list:
            best_list.append(payload)
            # 不做太复杂排序，长度控制一下即可
            if len(best_list) > max_events_per_id:
                self.id_best_payloads[cid] = best_list[-max_events_per_id:]
        else:
            self.id_best_payloads[cid] = best_list

    # ===================== payload 生成 =====================

    def _random_payload(self) -> bytes:
        """
        结构化随机生成 payload：
        - 50%：全随机 8 byte
        - 50%：稀疏激活（只动 1-3 个 byte，用典型值）
        同时插入一些“典型阈值附近的数值”，有利于触发 >=5555, <=32, <=96 等条件。
        """
        # 一些典型的 8-bit 和 16-bit 值
        interesting_byte_values = [0x00, 0x01, 0x02, 0x10, 0x20, 0x32, 0x40, 0x55, 0x5A, 0x60, 0x80, 0x96, 0xFF]
        interesting_word_values = [0x0000, 0x0101, 0x1010, 0x3232, 0x5555, 0x5A5A, 0x7FFF, 0xFFFF]

        if random.random() < 0.5:
            # 全随机
            return bytes(random.getrandbits(8) for _ in range(8))
        else:
            # 稀疏激活
            data = [0x00] * 8
            num_bytes = random.randint(1, 3)
            positions = random.sample(range(8), num_bytes)
            for pos in positions:
                if random.random() < 0.7:
                    data[pos] = random.choice(interesting_byte_values)
                else:
                    # 以 16-bit 方式赋值（跨两个 byte）
                    if pos <= 6:
                        val = random.choice(interesting_word_values)
                        data[pos] = (val & 0xFF)
                        data[pos + 1] = ((val >> 8) & 0xFF)
                    else:
                        data[pos] = random.choice(interesting_byte_values)
            return bytes(data)

    def _mutate_around(self, base: bytes) -> bytes:
        """
        在一个“已知有趣”的 payload 周围进行微调：
        - 翻转一些 bit 或对若干 byte 做 +delta/-delta。
        """
        data = list(base)
        # 随机选择 1~3 个字节进行变异
        for _ in range(random.randint(1, 3)):
            idx = random.randrange(8)
            mode = random.random()
            if mode < 0.5:
                # 翻转一个 bit
                bit = 1 << random.randint(0, 7)
                data[idx] ^= bit
            else:
                # 小幅加减
                delta = random.choice([-0x10, -0x08, -0x04, 0x04, 0x08, 0x10])
                data[idx] = (data[idx] + delta) & 0xFF
        return bytes(data)

    def _generate_mutated_payload(self, cid: int) -> bytes:
        """
        生成当前 ID 的 mutated payload：
        - 50% 纯随机/结构化随机；
        - 50% 在该 ID 已知高价值 payload 周围微调（如果有）。
        """
        use_guided = cid in self.id_best_payloads and self.id_best_payloads[cid] and random.random() < 0.5
        if use_guided:
            base = random.choice(self.id_best_payloads[cid])
            payload = self._mutate_around(base)
        else:
            payload = self._random_payload()

        # 确保 mutated 与 baseline 不同
        if payload == self.baseline_payload:
            payload = self._random_payload()

        return payload

    # ===================== bit→灯 映射构建（基于 baseline vs mutated） =====================

    def _build_bit_mapping_stats(self) -> List[Dict]:
        """
        对所有 single-type 试验，构建 bit→灯 映射统计：
        对每个 (ID, byte, bit, lamp) 统计：
        - 当 baseline→mutated 时该 bit 被改变的 episode 数；
        - 在这些 episode 中 lamp 0→1 / 1→0 的次数。
        """
        bit_stats: Dict[Tuple[int, int, int, int], Dict[str, int]] = {}
        bit_episode_counts: Dict[Tuple[int, int, int], int] = {}
        bit_example_payload: Dict[Tuple[int, int, int], str] = {}

        for rec in self.trial_log:
            if rec.get("type") != "single":
                continue

            cid = rec["id"]
            B_hex = rec.get("baseline_payload", "")
            M_hex = rec.get("mut_payload", "")
            if not B_hex or not M_hex:
                continue

            B = bytes.fromhex(B_hex)
            M = bytes.fromhex(M_hex)
            if len(B) != 8 or len(M) != 8:
                continue

            Lb_str = rec.get("L_before", "")
            La_str = rec.get("L_after", "")
            if not Lb_str or not La_str:
                continue
            L_before = [int(ch) for ch in Lb_str]
            L_after = [int(ch) for ch in La_str]
            if len(L_before) != len(L_after):
                continue

            # 该 episode 中哪些 bit 从 baseline 改变了
            mutated_bits: List[Tuple[int, int]] = []
            for bit_idx in range(64):
                byte_i = bit_idx // 8
                bit_i = bit_idx % 8
                b_bit = (B[byte_i] >> bit_i) & 0x1
                m_bit = (M[byte_i] >> bit_i) & 0x1
                if b_bit != m_bit:
                    mutated_bits.append((byte_i, bit_i))

            if not mutated_bits:
                continue

            # 记录 bit 被测试的次数 & 示例 payload
            for (byte_i, bit_i) in mutated_bits:
                key_bit = (cid, byte_i, bit_i)
                bit_episode_counts[key_bit] = bit_episode_counts.get(key_bit, 0) + 1
                if key_bit not in bit_example_payload:
                    bit_example_payload[key_bit] = M_hex

            # 根据灯的变化，把事件分派给所有 mutated bits
            for lamp_idx, (b, a) in enumerate(zip(L_before, L_after)):
                if b == a:
                    continue
                for (byte_i, bit_i) in mutated_bits:
                    key = (cid, byte_i, bit_i, lamp_idx)
                    st = bit_stats.get(key, {"on": 0, "off": 0})
                    if b == 0 and a == 1:
                        st["on"] += 1
                    elif b == 1 and a == 0:
                        st["off"] += 1
                    bit_stats[key] = st

        # 从统计中提取候选映射
        mappings: List[Dict] = []
        for (cid, byte_i, bit_i), ep_cnt in bit_episode_counts.items():
            if ep_cnt < self.min_bit_events_for_mapping:
                continue

            best_lamp = None
            best_on = 0
            best_off = 0
            best_total = 0

            for lamp_idx in range(self.K):
                st = bit_stats.get((cid, byte_i, bit_i, lamp_idx))
                if not st:
                    continue
                on = st["on"]
                off = st["off"]
                tot = on + off
                if tot > best_total:
                    best_total = tot
                    best_lamp = lamp_idx
                    best_on = on
                    best_off = off

            if best_lamp is None or best_total == 0:
                continue

            # 极性 & 置信度
            if best_on > 2 * best_off:
                polarity = "active_high"
                main_count = best_on
            elif best_off > 2 * best_on:
                polarity = "active_low"
                main_count = best_off
            else:
                polarity = "uncertain"
                main_count = max(best_on, best_off)

            confidence = main_count / float(ep_cnt)
            if confidence < self.min_confidence_for_mapping:
                continue

            label = self.labels[best_lamp] if 0 <= best_lamp < len(self.labels) else str(best_lamp)
            mappings.append({
                "id": cid,
                "id_hex": f"0x{cid:03X}",
                "byte_index": byte_i,
                "bit_index": bit_i,
                "lamp_index": best_lamp,
                "label": label,
                "polarity": polarity,
                "confidence": confidence,
                "on_count": best_on,
                "off_count": best_off,
                "episodes": ep_cnt,
                "example_payload": bit_example_payload.get((cid, byte_i, bit_i), ""),
            })

        mappings_sorted = sorted(mappings, key=lambda m: (m["id"], m["byte_index"], m["bit_index"]))
        return mappings_sorted
    # ===================== 灯级统计（用于概览和条形图） =====================

    def _compute_per_light_stats(self) -> List[Dict]:
        """
        统计每个 warning light 在所有 episode 中被点亮的次数，
        以及其参与的不同灯态模式数量（pattern_coverage）。
        """
        # 灯的总数以 labels 为准
        num_lights = len(self.labels)
        if num_lights == 0:
            return []

        trigger_counts = [0] * num_lights
        pattern_sets = [set() for _ in range(num_lights)]

        for rec in self.trial_log:
            L_after_str = rec.get("L_after", "")
            if not L_after_str:
                # 退化情况下仅用 lamp_on
                for idx in rec.get("lamp_on", []) or []:
                    if 0 <= idx < num_lights:
                        trigger_counts[idx] += 1
                continue

            try:
                L_after = tuple(int(ch) for ch in L_after_str)
            except Exception:
                continue

            # 统计“灯处于 ON 状态”的次数及其参与的 pattern
            for idx, v in enumerate(L_after):
                if idx >= num_lights:
                    break
                if v == 1:
                    trigger_counts[idx] += 1
                    pattern_sets[idx].add(L_after)

        stats = []
        for idx in range(num_lights):
            label = self.labels[idx] if idx < len(self.labels) else f"light_{idx}"
            stats.append({
                "index": idx,
                "label": label,
                "trigger_count": trigger_counts[idx],
                "pattern_coverage": len(pattern_sets[idx]),
            })
        return stats

    # ===================== 按灯的字段级候选信号（用于“按灯的候选映射表”） =====================

    def _build_signal_candidates_by_light(self) -> List[Dict]:
        """
        基于所有 single-type trial，按“灯 × ID”分析：
        - 优先识别 8-bit 等值字段：value == const
        - 其次尝试 8-bit 阈值字段：value <= T 或 value >= T
        - 再尝试 16-bit 阈值字段（little/big endian 两种）：

        输出为若干字段级候选：
        {
            "lamp_index": int,
            "lamp_label": str,
            "id": int,
            "id_hex": "0xXXX",
            "start_bit": int,
            "length": int,        # 8 or 16
            "endian": "na"/"little"/"big",
            "condition": str,     # "== 0x01", "<= 0x32", ">= 0x5555" 等
            "n_total": int,       # 该 (lamp, id) 所有 episode 数
            "n_pos": int,         # 其中该灯被点亮的 episode 数（0->1）
            "n_pos_cond": int,    # 满足条件且灯亮的 episode 数
            "n_neg_cond": int,    # 满足条件但灯没亮的 episode 数
            "p_pos_cond": float,  # 在灯亮样本中满足条件的比例
            "p_neg_cond": float,  # 在灯灭样本中满足条件的比例
            "field_type": str,    # "eq8", "le8", "ge8", "le16", "ge16"
        }
        """

        if not self.trial_log or not self.labels:
            return []

        num_lights = len(self.labels)
        min_on_samples = max(3, self.min_bit_events_for_mapping)  # 至少若干次正样本

        # 收集 (lamp, id) 维度上的样本：payload + 该灯是否在本 episode 中 0->1
        per_lamp_id_samples: Dict[Tuple[int, int], List[Tuple[bytes, bool]]] = defaultdict(list)

        for rec in self.trial_log:
            if rec.get("type") != "single":
                continue
            cid = rec.get("id")
            if cid is None:
                continue
            m_hex = rec.get("mut_payload", "")
            if not m_hex:
                continue
            try:
                payload = bytes.fromhex(m_hex)
            except ValueError:
                continue
            if len(payload) < 8:
                continue

            lamp_on_list = rec.get("lamp_on", []) or []
            for lamp_idx in range(num_lights):
                key = (lamp_idx, cid)
                flag_on = lamp_idx in lamp_on_list
                per_lamp_id_samples[key].append((payload, flag_on))

        # 一个 field 由 (lamp_idx, cid, start_bit, length, endian) 唯一确定
        field_map: Dict[Tuple[int, int, int, int, str], Dict] = {}

        # 阈值判定参数
        POS_THR = 0.9   # 在正样本中的覆盖率
        NEG_THR_EQ = 0.2  # 等值字段允许的假阳性比例
        NEG_THR_THR = 0.3  # 阈值字段允许的假阳性比例

        for (lamp_idx, cid), samples in per_lamp_id_samples.items():
            n_total = len(samples)
            if n_total == 0:
                continue

            # 拆分正负样本
            pos_payloads = [p for (p, flag) in samples if flag]
            neg_payloads = [p for (p, flag) in samples if not flag]
            n_pos = len(pos_payloads)
            n_neg = len(neg_payloads)
            if n_pos < min_on_samples:
                continue

            lamp_label = self.labels[lamp_idx] if lamp_idx < len(self.labels) else f"light_{lamp_idx}"

            # -------- 8-bit 字段分析：优先等值，其次阈值 --------
            for byte_idx in range(8):
                # 收集当前 byte 的正负样本值
                pos_vals = [p[byte_idx] for p in pos_payloads]
                neg_vals = [p[byte_idx] for p in neg_payloads]

                # 8-bit 等值：value == const
                eq_cand = None
                if pos_vals:
                    cnt_pos = Counter(pos_vals)
                    v_star, v_star_cnt = cnt_pos.most_common(1)[0]
                    p_on = v_star_cnt / len(pos_vals)
                    if n_neg > 0:
                        neg_hits = sum(1 for v in neg_vals if v == v_star)
                        p_off = neg_hits / n_neg
                    else:
                        neg_hits = 0
                        p_off = 0.0

                    if p_on >= POS_THR and p_off <= NEG_THR_EQ:
                        eq_cand = {
                            "lamp_index": lamp_idx,
                            "lamp_label": lamp_label,
                            "id": cid,
                            "id_hex": f"0x{cid:03X}",
                            "start_bit": byte_idx * 8,
                            "length": 8,
                            "endian": "na",
                            "condition": f"== 0x{v_star:02X}",
                            "n_total": n_total,
                            "n_pos": n_pos,
                            "n_pos_cond": v_star_cnt,
                            "n_neg_cond": neg_hits,
                            "p_pos_cond": p_on,
                            "p_neg_cond": p_off,
                            "field_type": "eq8",
                        }

                # 如果已经找到等值字段，就不再对这个 byte 做阈值分析
                if eq_cand is not None:
                    key = (lamp_idx, cid, eq_cand["start_bit"], eq_cand["length"], eq_cand["endian"])
                    old = field_map.get(key)
                    if (old is None) or (eq_cand["n_pos_cond"] > old.get("n_pos_cond", 0)):
                        field_map[key] = eq_cand
                    continue

                # 8-bit 阈值字段：value <= T or value >= T
                if not pos_vals or not neg_vals:
                    continue

                # <= T，取 T = 正样本中最大值（保证正样本基本都满足）
                T_le = max(pos_vals)
                pos_le = sum(1 for v in pos_vals if v <= T_le)
                neg_le = sum(1 for v in neg_vals if v <= T_le)
                p_pos_le = pos_le / len(pos_vals) if pos_vals else 0.0
                p_neg_le = neg_le / len(neg_vals) if neg_vals else 0.0

                best_thr_cand = None

                if p_pos_le >= POS_THR and p_neg_le <= NEG_THR_THR:
                    best_thr_cand = {
                        "lamp_index": lamp_idx,
                        "lamp_label": lamp_label,
                        "id": cid,
                        "id_hex": f"0x{cid:03X}",
                        "start_bit": byte_idx * 8,
                        "length": 8,
                        "endian": "na",
                            "condition": f"<= 0x{T_le:02X}",
                        "n_total": n_total,
                        "n_pos": n_pos,
                        "n_pos_cond": pos_le,
                        "n_neg_cond": neg_le,
                        "p_pos_cond": p_pos_le,
                        "p_neg_cond": p_neg_le,
                        "field_type": "le8",
                    }

                # >= T，取 T = 正样本中最小值
                T_ge = min(pos_vals)
                pos_ge = sum(1 for v in pos_vals if v >= T_ge)
                neg_ge = sum(1 for v in neg_vals if v >= T_ge)
                p_pos_ge = pos_ge / len(pos_vals) if pos_vals else 0.0
                p_neg_ge = neg_ge / len(neg_vals) if neg_vals else 0.0

                ge_cand = None
                if p_pos_ge >= POS_THR and p_neg_ge <= NEG_THR_THR:
                    ge_cand = {
                        "lamp_index": lamp_idx,
                        "lamp_label": lamp_label,
                        "id": cid,
                        "id_hex": f"0x{cid:03X}",
                        "start_bit": byte_idx * 8,
                        "length": 8,
                        "endian": "na",
                        "condition": f">= 0x{T_ge:02X}",
                        "n_total": n_total,
                        "n_pos": n_pos,
                        "n_pos_cond": pos_ge,
                        "n_neg_cond": neg_ge,
                        "p_pos_cond": p_pos_ge,
                        "p_neg_cond": p_neg_ge,
                        "field_type": "ge8",
                    }

                # 选择阈值方向：优先 p_pos 高，其次 p_neg 低
                cand = best_thr_cand
                if ge_cand is not None:
                    if (cand is None) or (
                        ge_cand["p_pos_cond"] > cand["p_pos_cond"] + 1e-6 or
                        (abs(ge_cand["p_pos_cond"] - cand["p_pos_cond"]) < 1e-6 and
                         ge_cand["p_neg_cond"] < cand["p_neg_cond"])
                    ):
                        cand = ge_cand

                if cand is not None:
                    key = (lamp_idx, cid, cand["start_bit"], cand["length"], cand["endian"])
                    old = field_map.get(key)
                    if (old is None) or (cand["n_pos_cond"] > old.get("n_pos_cond", 0)):
                        field_map[key] = cand

            # -------- 16-bit 阈值字段：little / big endian --------
            # 仅在至少两字节可用的情况下尝试
            for byte_idx in range(7):
                # 收集 16-bit 值
                pos_vals_le = [p[byte_idx] + 256 * p[byte_idx + 1] for p in pos_payloads]
                neg_vals_le = [p[byte_idx] + 256 * p[byte_idx + 1] for p in neg_payloads]
                pos_vals_be = [256 * p[byte_idx] + p[byte_idx + 1] for p in pos_payloads]
                neg_vals_be = [256 * p[byte_idx] + p[byte_idx + 1] for p in neg_payloads]

                def make_thr_candidate(vals_pos, vals_neg, endian: str) -> Optional[Dict]:
                    if not vals_pos:
                        return None
                    # <= T
                    T_le16 = max(vals_pos)
                    pos_le16 = sum(1 for v in vals_pos if v <= T_le16)
                    neg_le16 = sum(1 for v in vals_neg if v <= T_le16)
                    p_pos_le16 = pos_le16 / len(vals_pos) if vals_pos else 0.0
                    p_neg_le16 = neg_le16 / len(vals_neg) if vals_neg else 0.0

                    best_c = None
                    if p_pos_le16 >= POS_THR and p_neg_le16 <= NEG_THR_THR:
                        best_c = {
                            "lamp_index": lamp_idx,
                            "lamp_label": lamp_label,
                            "id": cid,
                            "id_hex": f"0x{cid:03X}",
                            "start_bit": byte_idx * 8,
                            "length": 16,
                            "endian": endian,
                            "condition": f"<= 0x{T_le16:04X}",
                            "n_total": n_total,
                            "n_pos": n_pos,
                            "n_pos_cond": pos_le16,
                            "n_neg_cond": neg_le16,
                            "p_pos_cond": p_pos_le16,
                            "p_neg_cond": p_neg_le16,
                            "field_type": "le16",
                        }

                    # >= T
                    T_ge16 = min(vals_pos)
                    pos_ge16 = sum(1 for v in vals_pos if v >= T_ge16)
                    neg_ge16 = sum(1 for v in vals_neg if v >= T_ge16)
                    p_pos_ge16 = pos_ge16 / len(vals_pos) if vals_pos else 0.0
                    p_neg_ge16 = neg_ge16 / len(vals_neg) if vals_neg else 0.0

                    ge_c = None
                    if p_pos_ge16 >= POS_THR and p_neg_ge16 <= NEG_THR_THR:
                        ge_c = {
                            "lamp_index": lamp_idx,
                            "lamp_label": lamp_label,
                            "id": cid,
                            "id_hex": f"0x{cid:03X}",
                            "start_bit": byte_idx * 8,
                            "length": 16,
                            "endian": endian,
                            "condition": f">= 0x{T_ge16:04X}",
                            "n_total": n_total,
                            "n_pos": n_pos,
                            "n_pos_cond": pos_ge16,
                            "n_neg_cond": neg_ge16,
                            "p_pos_cond": p_pos_ge16,
                            "p_neg_cond": p_neg_ge16,
                            "field_type": "ge16",
                        }

                    cand_local = best_c
                    if ge_c is not None:
                        if (cand_local is None) or (
                            ge_c["p_pos_cond"] > cand_local["p_pos_cond"] + 1e-6 or
                            (abs(ge_c["p_pos_cond"] - cand_local["p_pos_cond"]) < 1e-6 and
                             ge_c["p_neg_cond"] < cand_local["p_neg_cond"])
                        ):
                            cand_local = ge_c
                    return cand_local

                cand_le = make_thr_candidate(pos_vals_le, neg_vals_le, "little")
                cand_be = make_thr_candidate(pos_vals_be, neg_vals_be, "big")

                cand16 = cand_le
                if cand_be is not None:
                    if (cand16 is None) or (
                        cand_be["p_pos_cond"] > cand16["p_pos_cond"] + 1e-6 or
                        (abs(cand_be["p_pos_cond"] - cand16["p_pos_cond"]) < 1e-6 and
                         cand_be["p_neg_cond"] < cand16["p_neg_cond"])
                    ):
                        cand16 = cand_be

                if cand16 is not None:
                    key = (lamp_idx, cid, cand16["start_bit"], cand16["length"], cand16["endian"])
                    old = field_map.get(key)
                    if (old is None) or (cand16["n_pos_cond"] > old.get("n_pos_cond", 0)):
                        field_map[key] = cand16

        # 输出排序后的列表：按灯 index、ID、start_bit 排序
        candidates = list(field_map.values())
        candidates.sort(key=lambda c: (c["lamp_index"], c["id"], c["start_bit"], c["length"]))
        return candidates


    # ===================== 报告生成 =====================


    def _build_summary_dict(self) -> Dict:
        total_trials = len(self.trial_log)
        total_patterns = len(self.coverage)

        # 每个 ID 的整体统计
        id_summary = []
        for cid, stat in self.id_stats.items():
            if stat["N"] <= 0:
                continue
            patterns = self.id_patterns.get(cid, set())
            id_summary.append({
                "id": cid,
                "id_hex": f"0x{cid:03X}",
                "trials": int(stat["N"]),
                "mean_reward": float(stat["R"]),
                "pattern_count": len(patterns),
            })
        id_summary_sorted = sorted(id_summary, key=lambda x: x["mean_reward"], reverse=True)

        # bit 映射候选（按 bit 统计，用于低层调试）
        bit_mappings = self._build_bit_mapping_stats()

        # 每个灯的统计（用于概览 + 条形图）
        per_light_stats = self._compute_per_light_stats()

        # 字段级候选信号（按灯的候选映射表）
        signal_candidates = self._build_signal_candidates_by_light()

        # multi-ID 组合中 master warning 出现次数
        master_idx = self._find_master_warning_index()
        master_combo_hits = 0
        for rec in self.multi_combo_events:
            if rec.get("master_on"):
                master_combo_hits += 1

        summary = {
            "run_id": self.run_id,
            "global_id_range": [self.id_min, self.id_max],
            "total_trials": total_trials,
            "unique_patterns": total_patterns,
            "label_mapping": {i: lbl for i, lbl in enumerate(self.labels)},
            "id_summary": id_summary_sorted,
            "per_light_stats": per_light_stats,
            "signal_candidates": signal_candidates,
            "bit_mappings": bit_mappings,
            "multi_combo_count": len(self.multi_combo_events),
            "multi_combo_master_hits": master_combo_hits,
        }
        return summary

    def _write_text_report(self, summary: Dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("Vision-Guided Adaptive CAN Fuzzing Report\n")
            f.write(f"Run ID: {summary['run_id']}\n")
            f.write(
                f"Global ID range: 0x{summary['global_id_range'][0]:03X} "
                f"- 0x{summary['global_id_range'][1]:03X}\n"
            )
            f.write(f"Total trials: {summary['total_trials']}\n")
            f.write(f"Total unique light patterns: {summary['unique_patterns']}\n\n")

            f.write("Warning-light index mapping:\n")
            for idx, lbl in summary["label_mapping"].items():
                f.write(f"  [{idx}] {lbl}\n")
            f.write("\n")

            f.write("Per-ID statistics (sorted by mean_reward):\n")
            for entry in summary["id_summary"]:
                f.write(
                    f"- ID {entry['id_hex']}: "
                    f"trials={entry['trials']}, "
                    f"mean_reward={entry['mean_reward']:.2f}, "
                    f"pattern_count={entry['pattern_count']}\n"
                )
            f.write("\n")

            f.write(
                f"Multi-ID combo trials: {summary['multi_combo_count']}, "
                f"master warning hits: {summary['multi_combo_master_hits']}\n\n"
            )

            f.write("Candidate bit-to-warning-light mappings:\n\n")
            if not summary["bit_mappings"]:
                f.write("  (no candidates above thresholds)\n")
            else:
                for m in summary["bit_mappings"]:
                    f.write(
                        f"- ID {m['id_hex']} Byte {m['byte_index']} Bit {m['bit_index']} "
                        f"-> {m['label']} ({m['polarity']}, "
                        f"conf={m['confidence']:.2f}, "
                        f"on={m['on_count']}, off={m['off_count']}, "
                        f"episodes={m['episodes']}, "
                        f"example_payload={m['example_payload']})\n"
                    )

    def _write_pdf_dbc_table(self, summary: Dict, path: str):
        """
        用 ReportLab 输出 PDF，包含两张表：
        1) 每个 ID 的统计；
        2) bit-level 候选 DBC 表。
        """
        bit_mappings = summary.get("bit_mappings", [])
        id_summary = summary.get("id_summary", [])
        if not bit_mappings:
            print("[AdaptiveFuzzer] No bit mappings to export to PDF.")
            return

        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
        except Exception as e:
            print(f"[AdaptiveFuzzer] ReportLab not available, skip PDF export: {e}")
            return

        doc = SimpleDocTemplate(path, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        story = []

        title = Paragraph("Vision-Guided Candidate DBC Table", styles["Title"])
        info = Paragraph(f"Run ID: {summary['run_id']}", styles["Normal"])
        story.append(title)
        story.append(Spacer(1, 6))
        story.append(info)
        story.append(Spacer(1, 12))

        # 表 1：每个 ID 的统计
        if id_summary:
            data1 = [["ID (hex)", "Trials", "Mean reward", "#Patterns"]]
            for entry in id_summary:
                data1.append([
                    entry["id_hex"],
                    entry["trials"],
                    f"{entry['mean_reward']:.2f}",
                    entry["pattern_count"],
                ])
            table1 = Table(data1, repeatRows=1)
            style1 = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
            table1.setStyle(style1)
            story.append(table1)
            story.append(Spacer(1, 12))

        # 表 2：bit-level 候选 DBC 表
        data2 = [["ID (hex)", "Byte", "Bit", "Signal label", "Polarity",
                  "Confidence", "#On", "#Off", "#Episodes", "Example payload"]]
        for m in bit_mappings:
            data2.append([
                m["id_hex"],
                m["byte_index"],
                m["bit_index"],
                m["label"],
                m["polarity"],
                f"{m['confidence']:.2f}",
                m["on_count"],
                m["off_count"],
                m["episodes"],
                m["example_payload"],
            ])

        table2 = Table(data2, repeatRows=1)
        style2 = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
        table2.setStyle(style2)
        story.append(table2)

        doc.build(story)
        print(f"[AdaptiveFuzzer] DBC candidate PDF written to {path}")

    def _save_logs_and_report(self):
        """
        保存：
        - trial 级别日志 CSV；
        - summary.json；
        - summary.txt；
        - DBC 候选 PDF。
        """
        if not self.trial_log:
            print("[AdaptiveFuzzer] No trials recorded, skip logging.")
            return

        base = f"adaptive_fuzz_{self.run_id}"
        csv_path = os.path.join(self.log_dir, base + "_trials.csv")
        json_path = os.path.join(self.log_dir, base + "_summary.json")
        txt_path = os.path.join(self.log_dir, base + "_summary.txt")
        pdf_path = os.path.join(self.log_dir, base + "_dbc_candidates.pdf")

        # CSV
        fieldnames = sorted({k for rec in self.trial_log for k in rec.keys()})
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self.trial_log:
                    writer.writerow(rec)
            print(f"[AdaptiveFuzzer] Trial log written to {csv_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write CSV log: {e}")

        # summary
        summary = self._build_summary_dict()

        # JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[AdaptiveFuzzer] Summary JSON written to {json_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write JSON summary: {e}")

        # TXT
        try:
            self._write_text_report(summary, txt_path)
            print(f"[AdaptiveFuzzer] Summary text report written to {txt_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write text summary: {e}")

        # PDF
        try:
            self._write_pdf_dbc_table(summary, pdf_path)
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write PDF DBC table: {e}")
    def _write_pdf_full_report(self, summary: Dict, path: str):
        """
        生成完整的 PDF 报告，结构为：
        1. 运行概览：
           - Run ID、ID 范围、总 trials 数、总唯一灯光模式数
           - 每个灯被触发的次数 + pattern_coverage 的条形图和表
           - 每个 ID 的 mean_reward + pattern_count 排行表
        2. 按灯的候选信号表（字段级）：
           - Warning light, Lamp index, ID, StartBit, Length, Endian, Condition, 统计指标
        3. 高价值 ID 的 8 字节示意图：
           - 将有识别字段的字节高亮，其他字节灰色
           - 列出该 ID 的字段列表（起始 bit、长度、候选灯、条件/阈值）
        """
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate,
                Table,
                TableStyle,
                Paragraph,
                Spacer,
                PageBreak,
            )
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.graphics.shapes import Drawing, String, Rect
            from reportlab.graphics.charts.barcharts import VerticalBarChart
        except Exception as e:
            print(f"[AdaptiveFuzzer] ReportLab not available, skip PDF export: {e}")
            return

        id_summary = summary.get("id_summary", [])
        per_light_stats = summary.get("per_light_stats", [])
        signal_candidates = summary.get("signal_candidates", [])

        doc = SimpleDocTemplate(path, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        story = []

        # ========== Part 1: 运行概览 ==========
        title = Paragraph("Vision-Guided Adaptive CAN Fuzzing Report", styles["Title"])
        story.append(title)
        story.append(Spacer(1, 8))

        meta_lines = [
            f"Run ID: {summary.get('run_id', '')}",
            (
                "Global ID range: "
                f"0x{summary['global_id_range'][0]:03X} - "
                f"0x{summary['global_id_range'][1]:03X}"
                if summary.get("global_id_range") else ""
            ),
            f"Total trials: {summary.get('total_trials', 0)}",
            f"Total unique light patterns: {summary.get('unique_patterns', 0)}",
        ]
        for line in meta_lines:
            if line:
                story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))

        # ---- 1.1 每个灯触发次数 + coverage 条形图 ----
        if per_light_stats:
            story.append(Paragraph("Per-warning-light activation statistics", styles["Heading2"]))
            story.append(Spacer(1, 4))

            # 条形图：使用触发次数
            labels = [s["label"] for s in per_light_stats]
            counts = [int(s["trigger_count"]) for s in per_light_stats]
            if any(counts):
                short_labels = [
                    (lbl if len(lbl) <= 16 else (lbl[:15] + "…"))
                    for lbl in labels
                ]
                max_count = max(counts) or 1

                d = Drawing(500, 220)
                bc = VerticalBarChart()
                bc.x = 40
                bc.y = 40
                bc.width = 420
                bc.height = 140
                bc.data = [counts]
                bc.categoryAxis.categoryNames = short_labels
                bc.valueAxis.valueMin = 0
                bc.valueAxis.valueMax = max_count * 1.1
                bc.valueAxis.valueStep = max(1, max_count // 5 or 1)
                bc.categoryAxis.labels.boxAnchor = "ne"
                bc.categoryAxis.labels.angle = 45
                bc.categoryAxis.labels.dy = -30

                bc.bars.strokeColor = colors.black
                d.add(bc)
                story.append(d)
                story.append(Spacer(1, 8))

            # 同时给一张表：index, label, trigger_count, pattern_coverage
            data_light = [["Idx", "Warning light", "#Triggers", "Pattern coverage"]]
            for s in per_light_stats:
                data_light.append([
                    s["index"],
                    s["label"],
                    int(s["trigger_count"]),
                    int(s["pattern_coverage"]),
                ])
            table_light = Table(data_light, repeatRows=1)
            table_light.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(table_light)
            story.append(Spacer(1, 12))

        # ---- 1.2 每个 ID 的 mean_reward + pattern_count 排行 ----
        if id_summary:
            story.append(Paragraph("Per-ID statistics (sorted by mean reward)", styles["Heading2"]))
            story.append(Spacer(1, 4))

            data_id = [["ID (hex)", "Trials", "Mean reward", "#Patterns"]]
            for entry in id_summary:
                data_id.append([
                    entry["id_hex"],
                    entry["trials"],
                    f"{entry['mean_reward']:.2f}",
                    entry["pattern_count"],
                ])
            table_id = Table(data_id, repeatRows=1)
            table_id.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(table_id)
            story.append(Spacer(1, 12))

        # Multi-ID 组合概览
        story.append(Paragraph(
            f"Multi-ID combo trials: {summary.get('multi_combo_count', 0)}, "
            f"master warning hits: {summary.get('multi_combo_master_hits', 0)}",
            styles["Normal"]
        ))

        # ========== Part 2: 按灯的候选映射表 ==========
        story.append(PageBreak())
        story.append(Paragraph("Signal-level candidates by warning light", styles["Heading1"]))
        story.append(Spacer(1, 8))

        if not signal_candidates:
            story.append(Paragraph("No signal-level candidates above thresholds in this run.", styles["Normal"]))
        else:
            data_sig = [
                [
                    "Warning light",
                    "Lamp idx",
                    "ID (hex)",
                    "StartBit",
                    "Length",
                    "Endian",
                    "Condition",
                    "#Pos (lamp on)",
                    "#Neg (cond & lamp off)",
                    "Pos cov.",
                    "Neg rate",
                ]
            ]
            for c in signal_candidates:
                data_sig.append([
                    c.get("lamp_label", ""),
                    c.get("lamp_index", ""),
                    c.get("id_hex", ""),
                    c.get("start_bit", ""),
                    c.get("length", ""),
                    c.get("endian", ""),
                    c.get("condition", ""),
                    c.get("n_pos_cond", 0),
                    c.get("n_neg_cond", 0),
                    f"{c.get('p_pos_cond', 0.0):.2f}",
                    f"{c.get('p_neg_cond', 0.0):.2f}",
                ])

            table_sig = Table(data_sig, repeatRows=1)
            table_sig.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(table_sig)

        # ========== Part 3: 高价值 ID 的 8 字节示意图 ==========
        # 选择“高价值 ID”：所有出现在 signal_candidates 中的 ID
        story.append(PageBreak())
        story.append(Paragraph("Byte-level layout of high-value IDs", styles["Heading1"]))
        story.append(Spacer(1, 8))

        if signal_candidates:
            # 按 ID 聚合字段
            fields_by_id: Dict[int, List[Dict]] = defaultdict(list)
            for c in signal_candidates:
                cid = c.get("id")
                if cid is not None:
                    fields_by_id[cid].append(c)

            for cid in sorted(fields_by_id.keys()):
                cid_hex = f"0x{cid:03X}"
                story.append(Paragraph(f"ID {cid_hex} field layout (8 bytes)", styles["Heading2"]))
                story.append(Spacer(1, 4))

                fields = fields_by_id[cid]

                # 计算哪些 byte 被识别出字段覆盖
                recognized_bytes = set()
                for f in fields:
                    start_bit = int(f.get("start_bit", 0))
                    length = int(f.get("length", 0))
                    if start_bit % 8 != 0:
                        # 非字节对齐暂时标记为其所在字节
                        recognized_bytes.add(start_bit // 8)
                        continue
                    first_byte = start_bit // 8
                    if length <= 8:
                        recognized_bytes.add(first_byte)
                    else:
                        # 例如 16-bit 字段
                        recognized_bytes.add(first_byte)
                        if first_byte + 1 < 8:
                            recognized_bytes.add(first_byte + 1)

                # 画 8 个 byte 方块：识别出的用浅绿，未知用灰色
                d = Drawing(520, 100)
                margin_x = 20
                margin_y = 30
                box_w = 55
                box_h = 30

                for b_idx in range(8):
                    x = margin_x + b_idx * box_w
                    y = margin_y
                    if b_idx in recognized_bytes:
                        fill_color = colors.lightgreen
                    else:
                        fill_color = colors.lightgrey
                    d.add(Rect(x, y, box_w - 5, box_h, fillColor=fill_color, strokeColor=colors.black))
                    # 标注 byte index
                    d.add(String(x + 5, y + box_h / 2 - 4, f"B{b_idx}", fontSize=8))

                story.append(d)
                story.append(Spacer(1, 4))

                # 该 ID 下的字段列表：起始 bit、长度、灯、条件（含阈值）
                data_fields = [["StartBit", "Length", "Endian", "Warning light", "Condition", "Pos cov.", "Neg rate"]]
                # 为了可读性，按 start_bit、length 排序
                for f in sorted(fields, key=lambda ff: (ff.get("start_bit", 0), ff.get("length", 0))):
                    data_fields.append([
                        f.get("start_bit", ""),
                        f.get("length", ""),
                        f.get("endian", ""),
                        f.get("lamp_label", ""),
                        f.get("condition", ""),
                        f"{f.get('p_pos_cond', 0.0):.2f}",
                        f"{f.get('p_neg_cond', 0.0):.2f}",
                    ])

                table_fields = Table(data_fields, repeatRows=1)
                table_fields.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]))
                story.append(table_fields)
                story.append(Spacer(1, 12))

        doc.build(story)
        print(f"[AdaptiveFuzzer] Full PDF report written to {path}")



class SuperVisionGuidedAdaptiveFuzzer:
    """
    Vision-guided adaptive CAN fuzzer.

    关键特点：
    - ID 选择：全局覆盖 + epsilon-greedy bandit（带单 ID 最大尝试数）
    - 单 ID 试验：baseline payload vs mutated payload 成对对比
    - payload：结构化随机（全随机、稀疏激活、围绕高价值 payload 微调）
    - 时间窗口：mutated payload 默认连续发送多帧（有利于触发“3 次/2s”之类的条件）
    - bit→灯 映射：基于 baseline→mutated 过程中“变动位”和“灯变化”的统计
    - 多 ID 组合：周期性组合多个已知“有趣事件”来尝试触发 Master Warning 等多 ID 故障灯
    - 实验结束后输出：episodes.csv + summary.json + summary.txt + dbc_candidates.pdf
    """

    def __init__(
        self,
        can_fuzzer: CANFuzzer,
        detector: VisionDetector,
        id_start: int,
        id_end: int,
        seed_ids: Optional[List[int]] = None,
        # bandit & reward
        epsilon: float = 0.2,
        alpha: float = 1.0,
        beta: float = 5.0,
        # CAN 发送与时序
        default_freq_hz: float = 20.0,   # 建议稍微提一点频率
        baseline_repeats: int = 0,       # 后面会不再使用 baseline 帧
        mutated_repeats: int = 3,
        settle_time: float = 0.05,       # 现在当作 detection_delay 使用
        lamp_reset_time: float = 0.6,    # 新增：灯自动熄灭时间 + 安全裕量
        # ID 覆盖约束
        global_min_trials_per_id: int = 5,
        max_trials_per_id: int = 500,
        # 邻域扩展
        neighbor_delta: int = 1,
        neighbor_reward_threshold: float = 1.0,
        neighbor_min_trials: int = 10,
        # 多 ID 组合试验
        multi_combo_period: int = 50,
        min_events_for_combo: int = 3,
        # 视觉 warmup
        vision_warmup_time: float = 2.0,
        # bit→灯 映射阈值
        min_bit_events_for_mapping: int = 5,
        min_confidence_for_mapping: float = 0.6,
        # 日志目录
        log_dir: str = "logs",
    ):
        self.can_fuzzer = can_fuzzer
        self.detector = detector

        self.settle_time = float(settle_time)  # 现在把它当作 detection_delay
        self.lamp_reset_time = float(lamp_reset_time) # 两组报文之间的最小等待

        # ID 范围 & 初始候选集
        self.id_min = int(id_start)
        self.id_max = int(id_end)

        if seed_ids:
            self.id_candidates: List[int] = sorted(
                cid for cid in set(int(x) for x in seed_ids)
                if self.id_min <= cid <= self.id_max
            )
        else:
            self.id_candidates = list(range(self.id_min, self.id_max + 1))

        # 每个 ID 的 bandit 统计
        self.id_stats: Dict[int, Dict[str, float]] = {
            cid: {"R": 0.0, "N": 0.0} for cid in self.id_candidates
        }

        # 视觉 labels（YOLO 的类名）
        self.labels: List[str] = getattr(detector, "labels", [])
        self.K: int = len(self.labels)

        # baseline payload（全 0）
        self.baseline_payload: bytes = bytes([0x00] * 8)

        # 覆盖与日志
        self.coverage = set()  # set of tuples(light_state)
        self.id_patterns: Dict[int, set] = defaultdict(set)

        self.trial_log: List[Dict] = []  # 每个“试验”一条记录（single or multi）
        self.interesting_events: List[Dict] = []  # 单 ID 中发现有灯变化的“高价值事件”
        self.id_best_payloads: Dict[int, List[bytes]] = defaultdict(list)

        # multi-ID 组合统计（仅用于报告）
        self.multi_combo_events: List[Dict] = []

        # bandit & reward 参数
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # 发送与时间参数
        self.default_freq_hz = float(default_freq_hz)
        self.baseline_repeats = int(baseline_repeats)
        self.mutated_repeats = int(mutated_repeats)
        self.settle_time = float(settle_time)

        # 覆盖与 bandit 限制
        self.global_min_trials_per_id = int(global_min_trials_per_id)
        self.max_trials_per_id = int(max_trials_per_id)

        # 邻域扩展
        self.neighbor_delta = int(neighbor_delta)
        self.neighbor_reward_threshold = float(neighbor_reward_threshold)
        self.neighbor_min_trials = int(neighbor_min_trials)
        self.neighbor_expanded_ids = set()

        # 多 ID 组合参数
        self.multi_combo_period = int(multi_combo_period)
        self.min_events_for_combo = int(min_events_for_combo)

        # 视觉 warmup
        self.vision_warmup_time = float(vision_warmup_time)

        # bit 映射阈值
        self.min_bit_events_for_mapping = int(min_bit_events_for_mapping)
        self.min_confidence_for_mapping = float(min_confidence_for_mapping)

        # 日志路径
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._stop_flag = False

    # ===================== 公共接口 =====================

    def stop(self):
        """允许外部提前终止 run 循环。"""
        self._stop_flag = True

    def _wait_lamps_reset(self):
        """
        等待仪表盘上的灯自然熄灭。
        固定 sleep lamp_reset_time。
        也可以改成“循环检测直到所有灯都为 0 或超时”。
        """
        time.sleep(self.lamp_reset_time)


    def run(self, num_episodes: int = 5000):
        """
        主循环：执行 num_episodes 个“试验 episode”。
        每个 episode 要么是单 ID baseline vs mutated 试验，要么是多 ID 组合试验。
        """
        print(
            f"[AdaptiveFuzzer] Start fuzzing: {num_episodes} episodes, "
            f"ID range 0x{self.id_min:X}-0x{self.id_max:X}, "
            f"initial IDs={len(self.id_candidates)}"
        )

        self._warmup_vision()

        try:
            for ep in range(1, num_episodes + 1):
                if self._stop_flag:
                    print(f"[AdaptiveFuzzer] Stopped externally at episode {ep}.")
                    break

                # 周期性进行 multi-ID 组合尝试，用于触发 Master Warning 等
                if (
                    self.multi_combo_period > 0
                    and ep % self.multi_combo_period == 0
                    and len(self.interesting_events) >= self.min_events_for_combo
                ):
                    self._run_multi_id_combo_trial(ep)
                    continue

                # 常规：单 ID baseline vs mutated 试验
                self._run_single_id_trial(ep)

        finally:
            self._save_logs_and_report()
            print("[AdaptiveFuzzer] Fuzzing finished, logs & report generated.")

    # ===================== 视觉 warmup =====================

    def _warmup_vision(self):
        """在第一次发送 CAN 报文前预热摄像头和模型。"""
        if self.vision_warmup_time <= 0:
            return
        print(f"[AdaptiveFuzzer] Warming up vision for {self.vision_warmup_time:.1f} s...")
        end_t = time.time() + self.vision_warmup_time
        while time.time() < end_t:
            try:
                _ = self._get_light_state()
            except Exception as e:
                print(f"[AdaptiveFuzzer] Vision warmup error: {e}")
                break
            time.sleep(0.1)

    # ===================== 单 ID 试验（核心） =====================

    def _run_single_id_trial(self, ep: int):
        cid = self._select_id()

        # 1) 等待灯灭 + baseline 视觉
        self._wait_lamps_reset()
        L0 = self._get_light_state()  # 此时应该是全 0 或稳定状态

        # 2) mutated 报文：直接向该 ID 注入 payload
        M = self._generate_mutated_payload(cid)
        self._send_frames(cid, M, repeats=self.mutated_repeats)

        # 短暂等待，让仪表有时间刷新，但保证在 0.5s 内
        time.sleep(self.settle_time)
        L1 = self._get_light_state()

        # 3) reward & 灯变化
        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(L0, L1)

        # 4) 更新 bandit 和邻域
        self._update_bandit(cid, reward)
        self.id_patterns[cid].add(L1)
        self._maybe_expand_neighbors(cid)

        # 5) 记录有趣事件（用于 bit 映射与 multi-ID 组合）
        self._register_interesting_event(cid, M, reward, lamp_on, lamp_off)

        # 6) 日志记录
        rec = {
            "episode": ep,
            "type": "single",
            "id": cid,
            "id_hex": f"0x{cid:03X}",
            "baseline_payload": self.baseline_payload.hex(),  # 概念上的全 0 baseline
            "mut_payload": M.hex(),
            "reward": float(reward),
            "changed_cnt": int(changed_cnt),
            "is_new_pattern": bool(is_new),
            "L_before": "".join(str(x) for x in L0),
            "L_after": "".join(str(x) for x in L1),
            "lamp_on": lamp_on,
            "lamp_off": lamp_off,
        }
        self.trial_log.append(rec)

        print(
            f"[EP {ep:04d}] single ID=0x{cid:03X}, "
            f"mut={M.hex()}, Δlights={changed_cnt}, new={is_new}, "
            f"R_id={self.id_stats[cid]['R']:.2f}, N_id={int(self.id_stats[cid]['N'])}"
        )

    # ===================== 多 ID 组合试验（Master Warning 等） =====================

    def _run_multi_id_combo_trial(self, ep: int):
        if len(self.interesting_events) < self.min_events_for_combo:
            return

        events = random.sample(
            self.interesting_events,
            k=min(3, len(self.interesting_events))
        )

        # 1) 确保起点灯状态干净
        self._wait_lamps_reset()
        L_init = self._get_light_state()

        # 2) 快速依次发送多个 ID 的 payload
        for ev in events:
            cid = ev["id"]
            payload = ev["payload"]
            # 多 ID 组合时，为了不拉长时间，这里只发 1 帧
            self._send_frames(cid, payload, repeats=1, freq_hz=50.0)
            time.sleep(0.01)

        # 3) 短暂等待，保证在 0.5s 内检测
        time.sleep(self.settle_time)
        L_end = self._get_light_state()

        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(L_init, L_end)

        master_idx = self._find_master_warning_index()
        master_on = False
        if master_idx is not None and len(L_init) == len(L_end):
            if L_init[master_idx] == 0 and L_end[master_idx] == 1:
                master_on = True

        rec = {
            "episode": ep,
            "type": "multi",
            "id": -1,
            "id_hex": "multi",
            "sequence_ids": [ev["id"] for ev in events],
            "sequence_payloads": [ev["payload"].hex() for ev in events],
            "reward": float(reward),
            "changed_cnt": int(changed_cnt),
            "is_new_pattern": bool(is_new),
            "master_on": master_on,
            "L_before": "".join(str(x) for x in L_init),
            "L_after": "".join(str(x) for x in L_end),
            "lamp_on": lamp_on,
            "lamp_off": lamp_off,
        }
        self.trial_log.append(rec)
        self.multi_combo_events.append(rec)

        print(
            f"[EP {ep:04d}] multi-ID combo, "
            f"reward={reward:.2f}, Δlights={changed_cnt}, master_on={master_on}"
        )

    # ===================== ID 选择（bandit + 覆盖） =====================

    def _select_id(self) -> int:
        """
        两阶段：
        1) 覆盖：保证所有 ID 至少试验 global_min_trials_per_id 次；
        2) 在此基础上，对未达到 max_trials_per_id 的 ID 用 epsilon-greedy。
        """
        # 覆盖阶段
        min_N = min(stat["N"] for stat in self.id_stats.values())
        if min_N < self.global_min_trials_per_id:
            candidates = [cid for cid, stat in self.id_stats.items() if stat["N"] == min_N]
            return random.choice(candidates)

        # epsilon-greedy 阶段，限制单 ID 最大次数
        eligible_ids = [cid for cid, stat in self.id_stats.items() if stat["N"] < self.max_trials_per_id]
        if not eligible_ids:
            eligible_ids = list(self.id_stats.keys())

        if random.random() < self.epsilon:
            # 探索：在 eligible 中选 N 最小的
            min_N_elig = min(self.id_stats[cid]["N"] for cid in eligible_ids)
            candidates = [cid for cid in eligible_ids if self.id_stats[cid]["N"] == min_N_elig]
        else:
            # 利用：在 eligible 中选 R 最大的
            max_R = max(self.id_stats[cid]["R"] for cid in eligible_ids)
            candidates = [cid for cid in eligible_ids if self.id_stats[cid]["R"] == max_R]

        return random.choice(candidates)

    def _maybe_expand_neighbors(self, cid: int):
        """
        当某个 ID 的平均 reward 较高且尝试数达到一定阈值时，
        将其邻接 ID（cid±neighbor_delta）加入候选集。
        """
        if cid in self.neighbor_expanded_ids:
            return

        stat = self.id_stats[cid]
        if stat["N"] < self.neighbor_min_trials:
            return
        if stat["R"] < self.neighbor_reward_threshold:
            return

        new_ids = []
        for d in (-self.neighbor_delta, self.neighbor_delta):
            nid = cid + d
            if nid < self.id_min or nid > self.id_max:
                continue
            if nid in self.id_stats:
                continue
            new_ids.append(nid)

        if not new_ids:
            self.neighbor_expanded_ids.add(cid)
            return

        for nid in new_ids:
            self.id_stats[nid] = {"R": 0.0, "N": 0.0}
            self.id_candidates.append(nid)
            print(f"[AdaptiveFuzzer] Neighbor ID added: 0x{nid:03X}")

        self.neighbor_expanded_ids.add(cid)

    # ===================== CAN 发送 & 视觉采样 =====================

    def _send_frames(self, cid: int, payload: bytes, repeats: int = 1, freq_hz: Optional[float] = None):
        """
        在当前 ID 上重复发送若干帧 payload。
        freq_hz 为 None 时使用 default_freq_hz。
        """
        if freq_hz is None:
            freq_hz = self.default_freq_hz

        interval = 1.0 / float(freq_hz)
        bus: can.Bus = self.can_fuzzer.bus

        msg = can.Message(
            arbitration_id=cid,
            data=payload,
            is_extended_id=(cid > 0x7FF),
        )

        for _ in range(repeats):
            try:
                bus.send(msg)
            except can.CanError as e:
                print(f"[AdaptiveFuzzer] CAN send failed: {e}")
            time.sleep(interval)

    def _get_light_state(self) -> Tuple[int, ...]:
        """
        从视觉模块获取当前故障灯状态，映射为固定长度 0/1 向量。
        """
        detections = self.detector.detect()
        if not detections:
            return tuple([0] * self.K)

        if not self.labels:
            # 第一次检测时建立 labels 列表
            for det in detections:
                lbl = det["label"]
                if lbl not in self.labels:
                    self.labels.append(lbl)
            self.K = len(self.labels)

        label_to_idx = {lbl: idx for idx, lbl in enumerate(self.labels)}
        state = [0] * self.K

        for det in detections:
            lbl = det["label"]
            if lbl not in label_to_idx:
                # 出现新的 label，动态扩展
                self.labels.append(lbl)
                label_to_idx[lbl] = len(self.labels) - 1
                state.append(1)
                self.K = len(self.labels)
            else:
                state[label_to_idx[lbl]] = 1

        return tuple(state)

    def _find_master_warning_index(self) -> Optional[int]:
        """
        尝试在 labels 中找到 Master warning 灯的索引。
        规则：label 中包含 'Master' 或 'master'。
        """
        for idx, lbl in enumerate(self.labels):
            if "master" in lbl.lower():
                return idx
        return None

    # ===================== reward & event 统计 =====================

    def _compute_reward(
        self,
        L_before: Tuple[int, ...],
        L_after: Tuple[int, ...],
    ) -> Tuple[float, int, bool, List[int], List[int]]:
        """
        reward = alpha * (# 灯变化数) + beta * 1(出现新的灯态模式)
        同时返回哪些灯 0→1 / 1→0。
        """
        if len(L_before) != len(L_after):
            return 0.0, 0, False, [], []

        lamp_on = []
        lamp_off = []
        changed_cnt = 0

        for i, (b, a) in enumerate(zip(L_before, L_after)):
            if b != a:
                changed_cnt += 1
                if b == 0 and a == 1:
                    lamp_on.append(i)
                elif b == 1 and a == 0:
                    lamp_off.append(i)

        is_new = L_after not in self.coverage
        if is_new:
            self.coverage.add(L_after)

        reward = self.alpha * changed_cnt + (self.beta if is_new else 0.0)
        return reward, changed_cnt, is_new, lamp_on, lamp_off

    def _update_bandit(self, cid: int, reward: float):
        """增量更新该 ID 的平均 reward 和尝试次数。"""
        stat = self.id_stats[cid]
        N_old = stat["N"]
        R_old = stat["R"]
        N_new = N_old + 1.0
        R_new = R_old + (reward - R_old) / N_new
        stat["N"] = N_new
        stat["R"] = R_new

    def _register_interesting_event(
        self,
        cid: int,
        payload: bytes,
        reward: float,
        lamp_on: List[int],
        lamp_off: List[int],
        min_reward: float = 1.0,
        max_events_per_id: int = 20,
    ):
        """
        记录“有趣事件”：只要 reward 足够且发生了灯变化，用于：
        - multi-ID 组合实验
        - 后续（如需要）引导型 payload 生成
        """
        if reward < min_reward or not lamp_on:
            return

        labels = [self.labels[i] for i in lamp_on] if self.labels else []

        ev = {
            "id": cid,
            "payload": payload,
            "lamp_on": lamp_on,
            "lamp_off": lamp_off,
            "labels": labels,
            "reward": reward,
        }
        self.interesting_events.append(ev)
        if len(self.interesting_events) > 500:
            self.interesting_events = self.interesting_events[-500:]

        # 针对该 ID 保存若干高价值 payload
        best_list = self.id_best_payloads[cid]
        if payload not in best_list:
            best_list.append(payload)
            # 不做太复杂排序，长度控制一下即可
            if len(best_list) > max_events_per_id:
                self.id_best_payloads[cid] = best_list[-max_events_per_id:]
        else:
            self.id_best_payloads[cid] = best_list

    # ===================== payload 生成 =====================

    def _random_payload(self) -> bytes:
        """
        结构化随机生成 payload：
        - 50%：全随机 8 byte
        - 50%：稀疏激活（只动 1-3 个 byte，用典型值）
        同时插入一些“典型阈值附近的数值”，有利于触发 >=5555, <=32, <=96 等条件。
        """
        # 一些典型的 8-bit 和 16-bit 值
        interesting_byte_values = [0x00, 0x01, 0x02, 0x10, 0x20, 0x32, 0x40, 0x55, 0x5A, 0x60, 0x80, 0x96, 0xFF]
        interesting_word_values = [0x0000, 0x0101, 0x1010, 0x3232, 0x5555, 0x5A5A, 0x7FFF, 0xFFFF]

        if random.random() < 0.5:
            # 全随机
            return bytes(random.getrandbits(8) for _ in range(8))
        else:
            # 稀疏激活
            data = [0x00] * 8
            num_bytes = random.randint(1, 3)
            positions = random.sample(range(8), num_bytes)
            for pos in positions:
                if random.random() < 0.7:
                    data[pos] = random.choice(interesting_byte_values)
                else:
                    # 以 16-bit 方式赋值（跨两个 byte）
                    if pos <= 6:
                        val = random.choice(interesting_word_values)
                        data[pos] = (val & 0xFF)
                        data[pos + 1] = ((val >> 8) & 0xFF)
                    else:
                        data[pos] = random.choice(interesting_byte_values)
            return bytes(data)

    def _mutate_around(self, base: bytes) -> bytes:
        """
        在一个“已知有趣”的 payload 周围进行微调：
        - 翻转一些 bit 或对若干 byte 做 +delta/-delta。
        """
        data = list(base)
        # 随机选择 1~3 个字节进行变异
        for _ in range(random.randint(1, 3)):
            idx = random.randrange(8)
            mode = random.random()
            if mode < 0.5:
                # 翻转一个 bit
                bit = 1 << random.randint(0, 7)
                data[idx] ^= bit
            else:
                # 小幅加减
                delta = random.choice([-0x10, -0x08, -0x04, 0x04, 0x08, 0x10])
                data[idx] = (data[idx] + delta) & 0xFF
        return bytes(data)

    def _generate_mutated_payload(self, cid: int) -> bytes:
        """
        生成当前 ID 的 mutated payload：
        - 50% 纯随机/结构化随机；
        - 50% 在该 ID 已知高价值 payload 周围微调（如果有）。
        """
        use_guided = cid in self.id_best_payloads and self.id_best_payloads[cid] and random.random() < 0.5
        if use_guided:
            base = random.choice(self.id_best_payloads[cid])
            payload = self._mutate_around(base)
        else:
            payload = self._random_payload()

        # 确保 mutated 与 baseline 不同
        if payload == self.baseline_payload:
            payload = self._random_payload()

        return payload

    # ===================== bit→灯 映射构建（基于 baseline vs mutated） =====================

    def _build_bit_mapping_stats(self) -> List[Dict]:
        """
        对所有 single-type 试验，构建 bit→灯 映射统计：
        对每个 (ID, byte, bit, lamp) 统计：
        - 当 baseline→mutated 时该 bit 被改变的 episode 数；
        - 在这些 episode 中 lamp 0→1 / 1→0 的次数。
        """
        bit_stats: Dict[Tuple[int, int, int, int], Dict[str, int]] = {}
        bit_episode_counts: Dict[Tuple[int, int, int], int] = {}
        bit_example_payload: Dict[Tuple[int, int, int], str] = {}

        for rec in self.trial_log:
            if rec.get("type") != "single":
                continue

            cid = rec["id"]
            B_hex = rec.get("baseline_payload", "")
            M_hex = rec.get("mut_payload", "")
            if not B_hex or not M_hex:
                continue

            B = bytes.fromhex(B_hex)
            M = bytes.fromhex(M_hex)
            if len(B) != 8 or len(M) != 8:
                continue

            Lb_str = rec.get("L_before", "")
            La_str = rec.get("L_after", "")
            if not Lb_str or not La_str:
                continue
            L_before = [int(ch) for ch in Lb_str]
            L_after = [int(ch) for ch in La_str]
            if len(L_before) != len(L_after):
                continue

            # 该 episode 中哪些 bit 从 baseline 改变了
            mutated_bits: List[Tuple[int, int]] = []
            for bit_idx in range(64):
                byte_i = bit_idx // 8
                bit_i = bit_idx % 8
                b_bit = (B[byte_i] >> bit_i) & 0x1
                m_bit = (M[byte_i] >> bit_i) & 0x1
                if b_bit != m_bit:
                    mutated_bits.append((byte_i, bit_i))

            if not mutated_bits:
                continue

            # 记录 bit 被测试的次数 & 示例 payload
            for (byte_i, bit_i) in mutated_bits:
                key_bit = (cid, byte_i, bit_i)
                bit_episode_counts[key_bit] = bit_episode_counts.get(key_bit, 0) + 1
                if key_bit not in bit_example_payload:
                    bit_example_payload[key_bit] = M_hex

            # 根据灯的变化，把事件分派给所有 mutated bits
            for lamp_idx, (b, a) in enumerate(zip(L_before, L_after)):
                if b == a:
                    continue
                for (byte_i, bit_i) in mutated_bits:
                    key = (cid, byte_i, bit_i, lamp_idx)
                    st = bit_stats.get(key, {"on": 0, "off": 0})
                    if b == 0 and a == 1:
                        st["on"] += 1
                    elif b == 1 and a == 0:
                        st["off"] += 1
                    bit_stats[key] = st

        # 从统计中提取候选映射
        mappings: List[Dict] = []
        for (cid, byte_i, bit_i), ep_cnt in bit_episode_counts.items():
            if ep_cnt < self.min_bit_events_for_mapping:
                continue

            best_lamp = None
            best_on = 0
            best_off = 0
            best_total = 0

            for lamp_idx in range(self.K):
                st = bit_stats.get((cid, byte_i, bit_i, lamp_idx))
                if not st:
                    continue
                on = st["on"]
                off = st["off"]
                tot = on + off
                if tot > best_total:
                    best_total = tot
                    best_lamp = lamp_idx
                    best_on = on
                    best_off = off

            if best_lamp is None or best_total == 0:
                continue

            # 极性 & 置信度
            if best_on > 2 * best_off:
                polarity = "active_high"
                main_count = best_on
            elif best_off > 2 * best_on:
                polarity = "active_low"
                main_count = best_off
            else:
                polarity = "uncertain"
                main_count = max(best_on, best_off)

            confidence = main_count / float(ep_cnt)
            if confidence < self.min_confidence_for_mapping:
                continue

            label = self.labels[best_lamp] if 0 <= best_lamp < len(self.labels) else str(best_lamp)
            mappings.append({
                "id": cid,
                "id_hex": f"0x{cid:03X}",
                "byte_index": byte_i,
                "bit_index": bit_i,
                "lamp_index": best_lamp,
                "label": label,
                "polarity": polarity,
                "confidence": confidence,
                "on_count": best_on,
                "off_count": best_off,
                "episodes": ep_cnt,
                "example_payload": bit_example_payload.get((cid, byte_i, bit_i), ""),
            })

        mappings_sorted = sorted(mappings, key=lambda m: (m["id"], m["byte_index"], m["bit_index"]))
        return mappings_sorted

    # ===================== 报告生成 =====================

    def _build_summary_dict(self) -> Dict:
        total_trials = len(self.trial_log)
        total_patterns = len(self.coverage)

        # 每个 ID 的整体统计
        id_summary = []
        for cid, stat in self.id_stats.items():
            if stat["N"] <= 0:
                continue
            patterns = self.id_patterns.get(cid, set())
            id_summary.append({
                "id": cid,
                "id_hex": f"0x{cid:03X}",
                "trials": int(stat["N"]),
                "mean_reward": float(stat["R"]),
                "pattern_count": len(patterns),
            })
        id_summary_sorted = sorted(id_summary, key=lambda x: x["mean_reward"], reverse=True)

        # bit 映射候选
        bit_mappings = self._build_bit_mapping_stats()

        # multi-ID 组合中 master warning 出现次数
        master_idx = self._find_master_warning_index()
        master_combo_hits = 0
        for rec in self.multi_combo_events:
            if rec.get("master_on"):
                master_combo_hits += 1

        summary = {
            "run_id": self.run_id,
            "global_id_range": [self.id_min, self.id_max],
            "total_trials": total_trials,
            "unique_patterns": total_patterns,
            "label_mapping": {i: lbl for i, lbl in enumerate(self.labels)},
            "id_summary": id_summary_sorted,
            "bit_mappings": bit_mappings,
            "multi_combo_count": len(self.multi_combo_events),
            "multi_combo_master_hits": master_combo_hits,
        }
        return summary

    def _write_text_report(self, summary: Dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("Vision-Guided Adaptive CAN Fuzzing Report\n")
            f.write(f"Run ID: {summary['run_id']}\n")
            f.write(
                f"Global ID range: 0x{summary['global_id_range'][0]:03X} "
                f"- 0x{summary['global_id_range'][1]:03X}\n"
            )
            f.write(f"Total trials: {summary['total_trials']}\n")
            f.write(f"Total unique light patterns: {summary['unique_patterns']}\n\n")

            f.write("Warning-light index mapping:\n")
            for idx, lbl in summary["label_mapping"].items():
                f.write(f"  [{idx}] {lbl}\n")
            f.write("\n")

            f.write("Per-ID statistics (sorted by mean_reward):\n")
            for entry in summary["id_summary"]:
                f.write(
                    f"- ID {entry['id_hex']}: "
                    f"trials={entry['trials']}, "
                    f"mean_reward={entry['mean_reward']:.2f}, "
                    f"pattern_count={entry['pattern_count']}\n"
                )
            f.write("\n")

            f.write(
                f"Multi-ID combo trials: {summary['multi_combo_count']}, "
                f"master warning hits: {summary['multi_combo_master_hits']}\n\n"
            )

            f.write("Candidate bit-to-warning-light mappings:\n\n")
            if not summary["bit_mappings"]:
                f.write("  (no candidates above thresholds)\n")
            else:
                for m in summary["bit_mappings"]:
                    f.write(
                        f"- ID {m['id_hex']} Byte {m['byte_index']} Bit {m['bit_index']} "
                        f"-> {m['label']} ({m['polarity']}, "
                        f"conf={m['confidence']:.2f}, "
                        f"on={m['on_count']}, off={m['off_count']}, "
                        f"episodes={m['episodes']}, "
                        f"example_payload={m['example_payload']})\n"
                    )

    def _write_pdf_dbc_table(self, summary: Dict, path: str):
        """
        用 ReportLab 输出 PDF，包含两张表：
        1) 每个 ID 的统计；
        2) bit-level 候选 DBC 表。
        """
        bit_mappings = summary.get("bit_mappings", [])
        id_summary = summary.get("id_summary", [])
        if not bit_mappings:
            print("[AdaptiveFuzzer] No bit mappings to export to PDF.")
            return

        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
        except Exception as e:
            print(f"[AdaptiveFuzzer] ReportLab not available, skip PDF export: {e}")
            return

        doc = SimpleDocTemplate(path, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        story = []

        title = Paragraph("Vision-Guided Candidate DBC Table", styles["Title"])
        info = Paragraph(f"Run ID: {summary['run_id']}", styles["Normal"])
        story.append(title)
        story.append(Spacer(1, 6))
        story.append(info)
        story.append(Spacer(1, 12))

        # 表 1：每个 ID 的统计
        if id_summary:
            data1 = [["ID (hex)", "Trials", "Mean reward", "#Patterns"]]
            for entry in id_summary:
                data1.append([
                    entry["id_hex"],
                    entry["trials"],
                    f"{entry['mean_reward']:.2f}",
                    entry["pattern_count"],
                ])
            table1 = Table(data1, repeatRows=1)
            style1 = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
            table1.setStyle(style1)
            story.append(table1)
            story.append(Spacer(1, 12))

        # 表 2：bit-level 候选 DBC 表
        data2 = [["ID (hex)", "Byte", "Bit", "Signal label", "Polarity",
                  "Confidence", "#On", "#Off", "#Episodes", "Example payload"]]
        for m in bit_mappings:
            data2.append([
                m["id_hex"],
                m["byte_index"],
                m["bit_index"],
                m["label"],
                m["polarity"],
                f"{m['confidence']:.2f}",
                m["on_count"],
                m["off_count"],
                m["episodes"],
                m["example_payload"],
            ])

        table2 = Table(data2, repeatRows=1)
        style2 = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
        table2.setStyle(style2)
        story.append(table2)

        doc.build(story)
        print(f"[AdaptiveFuzzer] DBC candidate PDF written to {path}")

    def _save_logs_and_report(self):
        """
        保存：
        - trial 级别日志 CSV；
        - summary.json；
        - summary.txt；
        - DBC 候选 PDF。
        """
        if not self.trial_log:
            print("[AdaptiveFuzzer] No trials recorded, skip logging.")
            return

        base = f"adaptive_fuzz_{self.run_id}"
        csv_path = os.path.join(self.log_dir, base + "_trials.csv")
        json_path = os.path.join(self.log_dir, base + "_summary.json")
        txt_path = os.path.join(self.log_dir, base + "_summary.txt")
        pdf_path = os.path.join(self.log_dir, base + "_dbc_candidates.pdf")

        # CSV
        fieldnames = sorted({k for rec in self.trial_log for k in rec.keys()})
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self.trial_log:
                    writer.writerow(rec)
            print(f"[AdaptiveFuzzer] Trial log written to {csv_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write CSV log: {e}")

        # summary
        summary = self._build_summary_dict()

        # JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[AdaptiveFuzzer] Summary JSON written to {json_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write JSON summary: {e}")

        # TXT
        try:
            self._write_text_report(summary, txt_path)
            print(f"[AdaptiveFuzzer] Summary text report written to {txt_path}")
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write text summary: {e}")

        # PDF
        try:
            self._write_pdf_full_report(summary, pdf_path)
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write PDF report: {e}")

