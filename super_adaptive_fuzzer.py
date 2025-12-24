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
    """Vision-guided adaptive CAN fuzzer (super version).

    Generic design (no hard-coded ID semantics):
    - Online: use epsilon-greedy multi-armed bandit with vision feedback
      to choose IDs and payloads, maximizing warning-light coverage;
    - Support both single-ID and multi-ID / temporal conditions via:
      * per-episode lamp reset + detection within the 0.5 s window;
      * periodic multi-ID combo trials composed from prior high-reward events;
    - Offline: build statistical bit→light mappings and output a
      candidate DBC table, grouped by warning light, into a PDF report.

    The implementation does not special-case any particular CAN ID;
    everything is driven by statistics and bandit dynamics.
    """

    def __init__(
        self,
        can_fuzzer: CANFuzzer,
        detector: VisionDetector,
        id_start: int,
        id_end: int,
        # bandit & reward
        epsilon: float = 0.2,
        alpha: float = 1.0,
        beta: float = 5.0,
        # CAN sending & timing
        default_freq_hz: float = 20.0,
        baseline_repeats: int = 0,
        mutated_repeats: int = 3,
        settle_time: float = 0.1,
        lamp_reset_time: float = 0.6,
        # ID coverage constraints
        global_min_trials_per_id: int = 5,
        max_trials_per_id: int = 500,
        # neighbor expansion
        neighbor_delta: int = 1,
        neighbor_min_trials: int = 10,
        neighbor_reward_threshold: float = 1.0,
        # multi-ID combo
        multi_combo_period: int = 50,
        min_events_for_combo: int = 3,
        # vision warmup
        vision_warmup_time: float = 2.0,
        # bit→lamp mapping thresholds
        min_bit_events_for_mapping: int = 5,
        min_confidence_for_mapping: float = 0.6,
        # logging
        log_dir: str = "logsSuper",
    ):
        # core objects
        self.can_fuzzer = can_fuzzer
        self.detector = detector

        # ID range
        self.id_min = int(id_start)
        self.id_max = int(id_end)
        if self.id_min > self.id_max:
            self.id_min, self.id_max = self.id_max, self.id_min
        self.id_candidates: List[int] = list(range(self.id_min, self.id_max + 1))

        # per-ID bandit stats
        self.id_stats: Dict[int, Dict[str, float]] = {
            cid: {"R": 0.0, "N": 0.0} for cid in self.id_candidates
        }

        # vision labels (may grow dynamically)
        self.labels: List[str] = getattr(detector, "labels", [])
        self.K: int = len(self.labels)

        # conceptual baseline payload (not necessarily sent)
        self.baseline_payload: bytes = bytes([0x00] * 8)

        # coverage
        self.coverage = set()  # set of tuple(light_state)
        self.id_patterns: Dict[int, set] = defaultdict(set)

        # episode-level log & interesting events
        self.trial_log: List[Dict] = []
        self.interesting_events: List[Dict] = []
        self.id_best_payloads: Dict[int, List[bytes]] = defaultdict(list)
        self.multi_combo_events: List[Dict] = []

        # bandit & reward parameters
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # sending & timing
        self.default_freq_hz = float(default_freq_hz)
        self.baseline_repeats = int(baseline_repeats)
        self.mutated_repeats = int(mutated_repeats)
        # settle_time controls detection delay after sending payload
        self.settle_time = float(settle_time)
        # lamp_reset_time should be >= cluster auto-off (0.5 s) to avoid cross-episode interference
        self.lamp_reset_time = float(lamp_reset_time)

        # coverage & bandit limits
        self.global_min_trials_per_id = int(global_min_trials_per_id)
        self.max_trials_per_id = int(max_trials_per_id)

        # neighbor expansion
        self.neighbor_delta = int(neighbor_delta)
        self.neighbor_reward_threshold = float(neighbor_reward_threshold)
        self.neighbor_min_trials = int(neighbor_min_trials)
        self.neighbor_expanded_ids = set()

        # multi-ID combo
        self.multi_combo_period = int(multi_combo_period)
        self.min_events_for_combo = int(min_events_for_combo)

        # vision warmup
        self.vision_warmup_time = float(vision_warmup_time)

        # bit mapping thresholds
        self.min_bit_events_for_mapping = int(min_bit_events_for_mapping)
        self.min_confidence_for_mapping = float(min_confidence_for_mapping)

        # logging
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._stop_flag = False

    # ===================== public API =====================

    def stop(self):
        """Allow external code to stop the main loop."""
        self._stop_flag = True

    def run(self, num_episodes: int = 5000):
        """Main loop: run num_episodes single or multi-ID trials."""
        print(
            f"[SuperFuzzer] Start fuzzing: {num_episodes} episodes, "
            f"ID range 0x{self.id_min:X}-0x{self.id_max:X}"
        )

        self._warmup_vision()

        try:
            for ep in range(1, num_episodes + 1):
                if self._stop_flag:
                    print(f"[SuperFuzzer] Stopped externally at episode {ep}.")
                    break

                # periodic multi-ID combo to explore cross-ID / temporal semantics
                if (
                    self.multi_combo_period > 0
                    and ep % self.multi_combo_period == 0
                    and len(self.interesting_events) >= self.min_events_for_combo
                ):
                    self._run_multi_id_combo_trial(ep)
                else:
                    self._run_single_id_trial(ep)
        finally:
            self._save_logs_and_report()
            print("[SuperFuzzer] Fuzzing finished, logs & report generated.")

    # ===================== vision warmup & lamp reset =====================

    def _warmup_vision(self):
        if self.vision_warmup_time <= 0:
            return
        print(f"[SuperFuzzer] Warming up vision for {self.vision_warmup_time:.1f} s...")
        end_t = time.time() + self.vision_warmup_time
        while time.time() < end_t:
            try:
                _ = self._get_light_state()
            except Exception as exc:  # pragma: no cover
                print(f"[SuperFuzzer] Vision warmup error: {exc}")
                break
            time.sleep(0.1)

    def _wait_lamps_reset(self):
        """Wait for warning lights to naturally turn off."""
        time.sleep(self.lamp_reset_time)

    # ===================== single-ID trial =====================

    def _run_single_id_trial(self, ep: int):
        cid = self._select_id()

        # 1) reset lights and capture baseline state
        self._wait_lamps_reset()
        L0 = self._get_light_state()

        # 2) generate mutated payload and send frames for this ID
        M = self._generate_mutated_payload(cid)
        self._send_frames(cid, M, repeats=self.mutated_repeats)

        # ensure detection happens within ~0.5 s auto-off window
        time.sleep(self.settle_time)
        L1 = self._get_light_state()

        # 3) compute reward & light changes
        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(L0, L1)

        # 4) bandit update & neighbor expansion
        self._update_bandit(cid, reward)
        self.id_patterns[cid].add(L1)
        self._maybe_expand_neighbors(cid)

        # 5) register interesting event for later multi-ID combos & bit mapping
        self._register_interesting_event(ep, cid, M, reward, lamp_on, lamp_off)

        # 6) episode-level log
        rec = {
            "episode": ep,
            "type": "single",
            "id": cid,
            "id_hex": f"0x{cid:03X}",
            "baseline_payload": self.baseline_payload.hex(),
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
            f"[EP {ep:04d}] single ID=0x{cid:03X}, mut={M.hex()}, "
            f"Δlights={changed_cnt}, new={is_new}, "
            f"R_id={self.id_stats[cid]['R']:.2f}, N_id={int(self.id_stats[cid]['N'])}"
        )

    # ===================== multi-ID combo trial =====================

    def _run_multi_id_combo_trial(self, ep: int):
        """Run a multi-ID combo trial using previously interesting events."""
        if len(self.interesting_events) < self.min_events_for_combo:
            return

        # group events by warning-light labels (from vision)
        label_to_events: Dict[str, List[Dict]] = defaultdict(list)
        for ev in self.interesting_events:
            for lbl in ev.get("labels", []):
                label_to_events[lbl].append(ev)

        # count per-light occurrences to prefer rare lights in combos
        lamp_counts: Dict[str, int] = {
            lbl: len(evs) for lbl, evs in label_to_events.items()
        }
        sorted_labels = sorted(lamp_counts.keys(), key=lambda l: lamp_counts[l])

        # assemble up to 4 events, each from a different light if possible
        events: List[Dict] = []
        used_episodes = set()
        for lbl in sorted_labels:
            for ev in label_to_events[lbl]:
                ep_id = ev.get("episode")
                if ep_id in used_episodes:
                    continue
                events.append(ev)
                used_episodes.add(ep_id)
                break
            if len(events) >= 4:
                break

        if not events:
            events = random.sample(
                self.interesting_events,
                k=min(3, len(self.interesting_events)),
            )

        # 1) reset lights & capture baseline
        self._wait_lamps_reset()
        L_init = self._get_light_state()

        # 2) quickly send all selected ID/payload pairs
        for ev in events:
            cid = ev["id"]
            payload = ev["payload"]
            self._send_frames(cid, payload, repeats=1, freq_hz=50.0)
            time.sleep(0.01)

        # 3) detect within semantic window
        time.sleep(self.settle_time)
        L_end = self._get_light_state()

        reward, changed_cnt, is_new, lamp_on, lamp_off = self._compute_reward(
            L_init, L_end
        )

        # optional: try to detect master-like warning light (label contains 'master')
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
            f"[EP {ep:04d}] multi-ID combo, reward={reward:.2f}, "
            f"Δlights={changed_cnt}, master_on={master_on}"
        )

    # ===================== ID selection (bandit + coverage) =====================

    def _select_id(self) -> int:
        """Two-phase ID selection: coverage-first then epsilon-greedy bandit."""
        # coverage phase: ensure each ID has at least global_min_trials_per_id
        min_N = min(stat["N"] for stat in self.id_stats.values())
        if min_N < self.global_min_trials_per_id:
            candidates = [
                cid for cid, stat in self.id_stats.items() if stat["N"] == min_N
            ]
            return random.choice(candidates)

        # epsilon-greedy phase
        eligible_ids = [
            cid
            for cid, stat in self.id_stats.items()
            if stat["N"] < self.max_trials_per_id
        ]
        if not eligible_ids:
            # everyone hit the soft limit; pick among least-used IDs
            min_N_all = min(stat["N"] for stat in self.id_stats.values())
            eligible_ids = [
                cid for cid, stat in self.id_stats.items() if stat["N"] == min_N_all
            ]

        if random.random() < self.epsilon:
            # exploration: among eligible IDs, prefer least-tried ones
            min_N_elig = min(self.id_stats[cid]["N"] for cid in eligible_ids)
            candidates = [
                cid for cid in eligible_ids if self.id_stats[cid]["N"] == min_N_elig
            ]
        else:
            # exploitation: among eligible IDs, prefer highest mean reward
            max_R = max(self.id_stats[cid]["R"] for cid in eligible_ids)
            candidates = [
                cid for cid in eligible_ids if self.id_stats[cid]["R"] == max_R
            ]

        return random.choice(candidates)

    def _maybe_expand_neighbors(self, cid: int):
        """Generic neighbor expansion based on per-ID reward."""
        if cid in self.neighbor_expanded_ids:
            return

        stat = self.id_stats[cid]
        if stat["N"] < self.neighbor_min_trials:
            return
        if stat["R"] < self.neighbor_reward_threshold:
            return

        new_ids: List[int] = []
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
            print(f"[SuperFuzzer] Neighbor ID added: 0x{nid:03X}")

        self.neighbor_expanded_ids.add(cid)

    # ===================== CAN send & vision sampling =====================

    def _send_frames(
        self, cid: int, payload: bytes, repeats: int = 1, freq_hz: Optional[float] = None
    ):
        """Send a sequence of CAN frames on a given ID."""
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
            except can.CanError as exc:  # pragma: no cover
                print(f"[SuperFuzzer] CAN send failed: {exc}")
            time.sleep(interval)

    def _get_light_state(self) -> Tuple[int, ...]:
        """Query detector and convert detections to a fixed-length 0/1 vector."""
        detections = self.detector.detect()
        if not detections:
            if not self.labels:
                return tuple()
            return tuple([0] * self.K)

        # first detection: bootstrap label list
        if not self.labels:
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
                # new label discovered at runtime
                self.labels.append(lbl)
                label_to_idx[lbl] = len(self.labels) - 1
                state.append(1)
                self.K = len(self.labels)
            else:
                state[label_to_idx[lbl]] = 1

        return tuple(state)

    def _find_master_warning_index(self) -> Optional[int]:
        """Try to locate a master-warning-like lamp by label substring."""
        for idx, lbl in enumerate(self.labels):
            if "master" in lbl.lower():
                return idx
        return None

    # ===================== reward & event registration =====================

    def _compute_reward(
        self,
        L_before: Tuple[int, ...],
        L_after: Tuple[int, ...],
    ) -> Tuple[float, int, bool, List[int], List[int]]:
        """Compute reward based on light-state changes."""
        if len(L_before) != len(L_after):
            return 0.0, 0, False, [], []

        lamp_on: List[int] = []
        lamp_off: List[int] = []
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
        """Incrementally update mean reward for the given ID."""
        stat = self.id_stats[cid]
        N_old = stat["N"]
        R_old = stat["R"]
        N_new = N_old + 1.0
        R_new = R_old + (reward - R_old) / N_new
        stat["N"] = N_new
        stat["R"] = R_new

    def _register_interesting_event(
        self,
        ep: int,
        cid: int,
        payload: bytes,
        reward: float,
        lamp_on: List[int],
        lamp_off: List[int],
        min_reward: float = 1.0,
        max_events_per_id: int = 50,
    ) -> None:
        """Record high-reward events that changed at least one light."""
        if reward < min_reward or not lamp_on:
            return

        labels = [self.labels[i] for i in lamp_on] if self.labels else []
        ev = {
            "episode": ep,
            "id": cid,
            "payload": payload,
            "lamp_on": lamp_on,
            "lamp_off": lamp_off,
            "labels": labels,
            "reward": reward,
        }
        self.interesting_events.append(ev)
        if len(self.interesting_events) > 2000:
            self.interesting_events = self.interesting_events[-2000:]

        # maintain a small buffer of best payloads per ID for guided mutation
        best_list = self.id_best_payloads[cid]
        if payload not in best_list:
            best_list.append(payload)
            if len(best_list) > max_events_per_id:
                self.id_best_payloads[cid] = best_list[-max_events_per_id:]
        else:
            self.id_best_payloads[cid] = best_list

    # ===================== payload generation =====================

    def _random_payload(self) -> bytes:
        """Structured random payload generation."""
        mode = random.random()
        if mode < 0.5:
            # fully random 8-byte payload
            return bytes(random.getrandbits(8) for _ in range(8))

        # sparse / structured payload
        data = [0x00] * 8
        num_bytes = random.randint(1, 3)
        byte_positions = random.sample(range(8), num_bytes)

        # include 0x01, 0x10, 0x20... to better excite individual bits/thresholds
        interesting_byte_values = [
            0x00,
            0x01,
            0x02,
            0x04,
            0x08,
            0x10,
            0x20,
            0x32,
            0x40,
            0x55,
            0x5A,
            0x60,
            0x80,
            0x96,
            0xFF,
        ]
        interesting_word_values = [
            0x0000,
            0x0101,
            0x1010,
            0x3232,
            0x5555,
            0x5A5A,
            0x7FFF,
            0xFFFF,
        ]

        for pos in byte_positions:
            if random.random() < 0.7:
                data[pos] = random.choice(interesting_byte_values)
            else:
                if pos <= 6:
                    val = random.choice(interesting_word_values)
                    data[pos] = val & 0xFF
                    data[pos + 1] = (val >> 8) & 0xFF
                else:
                    data[pos] = random.choice(interesting_byte_values)

        return bytes(data)

    def _mutate_around(self, base: bytes) -> bytes:
        """Local mutation around a known interesting payload."""
        data = list(base)
        # mutate 1–3 bytes
        for _ in range(random.randint(1, 3)):
            idx = random.randrange(8)
            mode = random.random()
            if mode < 0.5:
                # flip a random bit
                bit = 1 << random.randint(0, 7)
                data[idx] ^= bit
            else:
                # small +/- delta on the whole byte
                delta = random.randint(-8, 8)
                data[idx] = (data[idx] + delta) & 0xFF
        return bytes(data)

    def _generate_mutated_payload(self, cid: int) -> bytes:
        """Generate a mutated payload for the given ID."""
        use_guided = (
            cid in self.id_best_payloads
            and self.id_best_payloads[cid]
            and random.random() < 0.5
        )
        if use_guided:
            base = random.choice(self.id_best_payloads[cid])
            payload = self._mutate_around(base)
        else:
            payload = self._random_payload()

        if payload == self.baseline_payload:
            payload = self._random_payload()

        return payload

    # ===================== bit→lamp mapping (offline inference) =====================

    def _build_bit_mapping_stats(self) -> List[Dict]:
        """Build bit→warning-light mapping candidates from trial_log."""
        if not self.trial_log or not self.labels:
            return []

        # bit_stats[(id, byte_idx, bit_idx, lamp_idx)] = {...}
        bit_stats: Dict[Tuple[int, int, int, int], Dict[str, object]] = {}

        for rec in self.trial_log:
            if rec.get("type") != "single":
                continue

            cid = rec.get("id")
            M_hex = rec.get("mut_payload")
            if cid is None or not M_hex:
                continue
            try:
                payload = bytes.fromhex(M_hex)
            except ValueError:
                continue
            if len(payload) < 1:
                continue

            L_after_str = rec.get("L_after", "")
            if not L_after_str:
                continue
            L_after = [int(ch) for ch in L_after_str]
            lamp_count = len(L_after)
            if lamp_count == 0:
                continue

            total_bits = len(payload) * 8
            for bit_pos in range(total_bits):
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                bit_val = (payload[byte_idx] >> bit_idx) & 0x1

                for lamp_idx in range(lamp_count):
                    lamp_state = L_after[lamp_idx]
                    key = (cid, byte_idx, bit_idx, lamp_idx)
                    stats = bit_stats.get(
                        key,
                        {
                            "bit1_on": 0,
                            "bit1_total": 0,
                            "bit0_on": 0,
                            "bit0_total": 0,
                            "example1": None,
                            "example0": None,
                        },
                    )

                    if bit_val == 1:
                        stats["bit1_total"] = int(stats["bit1_total"]) + 1
                        if lamp_state == 1:
                            stats["bit1_on"] = int(stats["bit1_on"]) + 1
                            if stats["example1"] is None:
                                stats["example1"] = M_hex
                    else:
                        stats["bit0_total"] = int(stats["bit0_total"]) + 1
                        if lamp_state == 1:
                            stats["bit0_on"] = int(stats["bit0_on"]) + 1
                            if stats["example0"] is None:
                                stats["example0"] = M_hex

                    bit_stats[key] = stats

        mappings: List[Dict] = []
        for (cid, byte_idx, bit_idx, lamp_idx), st in bit_stats.items():
            bit1_on = int(st["bit1_on"])
            bit1_total = int(st["bit1_total"])
            bit0_on = int(st["bit0_on"])
            bit0_total = int(st["bit0_total"])

            total_samples = bit1_total + bit0_total
            lamp_on_total = bit1_on + bit0_on

            if lamp_on_total < self.min_bit_events_for_mapping:
                continue
            if bit1_total == 0 or bit0_total == 0:
                continue

            # rule 1: active_high (bit=1 -> lamp=1)
            TP = bit1_on
            FP = bit1_total - bit1_on
            FN = bit0_on
            denom_high = 2 * TP + FP + FN
            F1_high = 0.0 if denom_high == 0 else (2.0 * TP) / denom_high

            # rule 2: active_low (bit=0 -> lamp=1)
            TP_low = bit0_on
            FP_low = bit0_total - bit0_on
            FN_low = bit1_on
            denom_low = 2 * TP_low + FP_low + FN_low
            F1_low = 0.0 if denom_low == 0 else (2.0 * TP_low) / denom_low

            if F1_high <= 0.0 and F1_low <= 0.0:
                continue

            if F1_high >= F1_low:
                polarity = "active_high"
                confidence = F1_high
                example_payload = st["example1"]
            else:
                polarity = "active_low"
                confidence = F1_low
                example_payload = st["example0"]

            if confidence < self.min_confidence_for_mapping:
                continue

            label = (
                self.labels[lamp_idx]
                if 0 <= lamp_idx < len(self.labels)
                else f"lamp_{lamp_idx}"
            )

            mappings.append(
                {
                    "id": cid,
                    "id_hex": f"0x{cid:03X}",
                    "byte_index": byte_idx,
                    "bit_index": bit_idx,
                    "lamp_index": lamp_idx,
                    "label": label,
                    "polarity": polarity,
                    "confidence": float(confidence),
                    "bit1_on": bit1_on,
                    "bit1_total": bit1_total,
                    "bit0_on": bit0_on,
                    "bit0_total": bit0_total,
                    "lamp_on_total": lamp_on_total,
                    "sample_total": total_samples,
                    "example_payload": example_payload or "",
                }
            )

        mappings_sorted = sorted(
            mappings,
            key=lambda m: (m["id"], m["byte_index"], m["bit_index"], m["lamp_index"]),
        )
        return mappings_sorted

    # ===================== summary & reporting =====================

    def _build_summary_dict(self) -> Dict:
        total_trials = len(self.trial_log)
        total_patterns = len(self.coverage)

        id_summary = []
        for cid, stat in self.id_stats.items():
            if stat["N"] <= 0:
                continue
            patterns = self.id_patterns.get(cid, set())
            id_summary.append(
                {
                    "id": cid,
                    "id_hex": f"0x{cid:03X}",
                    "trials": int(stat["N"]),
                    "mean_reward": float(stat["R"]),
                    "pattern_count": len(patterns),
                }
            )

        id_summary_sorted = sorted(
            id_summary, key=lambda x: x["mean_reward"], reverse=True
        )

        bit_mappings = self._build_bit_mapping_stats()

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
            "master_index": master_idx,
        }
        return summary

    def _write_text_report(self, summary: Dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("Vision-Guided Adaptive CAN Fuzzing Report (Super Version)\n")
            f.write(f"Run ID: {summary['run_id']}\n")
            f.write(
                f"Global ID range: 0x{summary['global_id_range'][0]:03X} "
                f"- 0x{summary['global_id_range'][1]:03X}\n"
            )
            f.write(f"Total trials: {summary['total_trials']}\n")
            f.write(
                f"Total unique light patterns: {summary['unique_patterns']}\n\n"
            )

            f.write("Warning-light index mapping (YOLO labels):\n")
            for idx, lbl in summary["label_mapping"].items():
                f.write(f"  [{idx}] {lbl}\n")
            f.write("\n")

            f.write("Per-ID statistics (sorted by mean_reward):\n")
            for entry in summary["id_summary"]:
                f.write(
                    f"- ID {entry['id_hex']}: trials={entry['trials']}, "
                    f"mean_reward={entry['mean_reward']:.2f}, "
                    f"pattern_count={entry['pattern_count']}\n"
                )
            f.write("\n")

            f.write(
                f"Multi-ID combo trials: {summary['multi_combo_count']}, "
                f"master-like warning hits: {summary['multi_combo_master_hits']}\n\n"
            )

            f.write("Candidate bit-to-warning-light mappings (thresholded):\n\n")
            if not summary["bit_mappings"]:
                f.write("  (no candidates above thresholds)\n")
            else:
                for m in summary["bit_mappings"]:
                    f.write(
                        f"- {m['label']} :: ID {m['id_hex']} "
                        f"Byte {m['byte_index']} Bit {m['bit_index']} "
                        f"({m['polarity']}, conf={m['confidence']:.2f}, "
                        f"lamp_on={m['lamp_on_total']}, "
                        f"samples={m['sample_total']}, "
                        f"example_payload={m['example_payload']})\n"
                    )

    def _write_pdf_dbc_table(self, summary: Dict, path: str) -> None:
        """Write ID-level summary and bit→lamp candidate table into a PDF."""
        bit_mappings = summary.get("bit_mappings", [])
        id_summary = summary.get("id_summary", [])
        if not bit_mappings and not id_summary:
            print("[SuperFuzzer] No data to export to PDF.")
            return

        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate,
                Table,
                TableStyle,
                Paragraph,
                Spacer,
            )
            from reportlab.lib.styles import getSampleStyleSheet
        except Exception as exc:  # pragma: no cover
            print(f"[SuperFuzzer] ReportLab not available, skip PDF export: {exc}")
            return

        doc = SimpleDocTemplate(path, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        story = []

        title = Paragraph(
            "Vision-Guided Candidate DBC Table (Bit-to-Warning-Light Mapping)",
            styles["Title"],
        )
        info = Paragraph(f"Run ID: {summary['run_id']}", styles["Normal"])
        story.append(title)
        story.append(Spacer(1, 6))
        story.append(info)
        story.append(Spacer(1, 12))

        # Table 1: per-ID summary
        if id_summary:
            data1 = [["ID (hex)", "Trials", "Mean reward", "#Patterns"]]
            for entry in id_summary:
                data1.append(
                    [
                        entry["id_hex"],
                        entry["trials"],
                        f"{entry['mean_reward']:.2f}",
                        entry["pattern_count"],
                    ]
                )
            table1 = Table(data1, repeatRows=1)
            style1 = TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
            table1.setStyle(style1)
            story.append(table1)
            story.append(Spacer(1, 12))

        # Table 2: bit→lamp candidate DBC (grouped by warning light)
        if bit_mappings:
            sorted_bits = sorted(
                bit_mappings,
                key=lambda m: (
                    m["label"],
                    m["id_hex"],
                    m["byte_index"],
                    m["bit_index"],
                ),
            )

            data2 = [
                [
                    "Warning light",
                    "ID (hex)",
                    "Byte",
                    "Bit",
                    "Polarity",
                    "Confidence",
                    "#LampOn",
                    "#Bit=1 & LampOn",
                    "#Bit=1 total",
                    "#Bit=0 & LampOn",
                    "#Bit=0 total",
                    "Example payload",
                ]
            ]

            for m in sorted_bits:
                data2.append(
                    [
                        m["label"],
                        m["id_hex"],
                        m["byte_index"],
                        m["bit_index"],
                        m["polarity"],
                        f"{m['confidence']:.2f}",
                        m["lamp_on_total"],
                        m["bit1_on"],
                        m["bit1_total"],
                        m["bit0_on"],
                        m["bit0_total"],
                        m["example_payload"],
                    ]
                )

            table2 = Table(data2, repeatRows=1)
            style2 = TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
            table2.setStyle(style2)
            story.append(table2)

        doc.build(story)
        print(f"[SuperFuzzer] DBC candidate PDF written to {path}")

    def _save_logs_and_report(self) -> None:
        """Dump per-trial CSV, JSON summary, text report, and PDF DBC table."""
        if not self.trial_log:
            print("[SuperFuzzer] No trials recorded, skip logging.")
            return

        base = f"adaptive_fuzz_{self.run_id}"
        csv_path = os.path.join(self.log_dir, base + "_trials.csv")
        json_path = os.path.join(self.log_dir, base + "_summary.json")
        txt_path = os.path.join(self.log_dir, base + "_summary.txt")
        pdf_path = os.path.join(self.log_dir, base + "_dbc_candidates.pdf")

        # CSV trial log
        fieldnames = sorted({k for rec in self.trial_log for k in rec.keys()})
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self.trial_log:
                    writer.writerow(rec)
            print(f"[SuperFuzzer] Trial log written to {csv_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[SuperFuzzer] Failed to write CSV log: {exc}")

        summary = self._build_summary_dict()

        # JSON summary
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[SuperFuzzer] Summary JSON written to {json_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[SuperFuzzer] Failed to write JSON summary: {exc}")

        # text report
        try:
            self._write_text_report(summary, txt_path)
            print(f"[SuperFuzzer] Summary text report written to {txt_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[SuperFuzzer] Failed to write text summary: {exc}")

        # PDF report
        try:
            self._write_pdf_dbc_table(summary, pdf_path)
        except Exception as exc:  # pragma: no cover
            print(f"[SuperFuzzer] Failed to write PDF DBC table: {exc}")
