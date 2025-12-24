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
        default_freq_hz: float = 10.0,
        baseline_repeats: int = 1,
        mutated_repeats: int = 3,
        settle_time: float = 0.2,
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

        # 1) baseline：全 0 payload
        B = self.baseline_payload
        self._send_frames(cid, B, repeats=self.baseline_repeats)
        time.sleep(self.settle_time)
        L0 = self._get_light_state()

        # 2) mutated：结构化随机 payload
        M = self._generate_mutated_payload(cid)
        self._send_frames(cid, M, repeats=self.mutated_repeats)
        time.sleep(self.settle_time)
        L1 = self._get_light_state()

        # 3) reward & 灯变化统计
        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(L0, L1)

        # 4) 更新 bandit
        self._update_bandit(cid, reward)
        self.id_patterns[cid].add(L1)
        self._maybe_expand_neighbors(cid)

        # 5) 如果灯有变化且 reward 足够，记录为“有趣事件”，用于后续组合 & 引导
        self._register_interesting_event(cid, M, reward, lamp_on, lamp_off)

        # 6) 记录日志
        rec = {
            "episode": ep,
            "type": "single",
            "id": cid,
            "id_hex": f"0x{cid:03X}",
            "baseline_payload": B.hex(),
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

        # 控制台打印
        print(
            f"[EP {ep:04d}] single ID=0x{cid:03X}, "
            f"mut={M.hex()}, Δlights={changed_cnt}, new={is_new}, "
            f"R_id={self.id_stats[cid]['R']:.2f}, N_id={int(self.id_stats[cid]['N'])}"
        )

    # ===================== 多 ID 组合试验（Master Warning 等） =====================

    def _run_multi_id_combo_trial(self, ep: int):
        """
        组合多个已发现“有灯变化”的事件，尝试触发 Master Warning 等多 ID 灯。
        不参与 bandit 更新，只用于探索高阶语义。
        """
        # 选取若干最近的有趣事件
        if len(self.interesting_events) < self.min_events_for_combo:
            return

        events = random.sample(self.interesting_events, k=min(3, len(self.interesting_events)))

        # 记录初始灯态
        time.sleep(self.settle_time)
        L_init = self._get_light_state()

        # 依次发送各事件的 payload
        for ev in events:
            cid = ev["id"]
            payload = ev["payload"]
            self._send_frames(cid, payload, repeats=self.mutated_repeats)
            time.sleep(0.2)  # ID 之间的小间隔

        time.sleep(self.settle_time)
        L_end = self._get_light_state()

        # 判断整体灯变化
        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(L_init, L_end)

        # 专门统计 master warning 是否被点亮
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

    def _send_frames(self, cid: int, payload: bytes, repeats: int = 1):
        """
        在当前 ID 上重复发送若干帧 payload。
        """
        interval = 1.0 / self.default_freq_hz
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
            self._write_pdf_dbc_table(summary, pdf_path)
        except Exception as e:
            print(f"[AdaptiveFuzzer] Failed to write PDF DBC table: {e}")
