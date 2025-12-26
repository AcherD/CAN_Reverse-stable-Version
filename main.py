import threading
import time
from multiprocessing import Process, Manager
from VisionDetector import VisionDetector
from CAN_fuzzer import CANFuzzer
from CVAutoFuzz import CVAutoFuzz
import yaml
from pathlib import Path
from queue import Queue
import os
from adaptive_fuzzer import VisionGuidedAdaptiveFuzzer
from super_adaptive_fuzzer import SuperVisionGuidedAdaptiveFuzzer

def load_yaml(yaml_path: str) -> dict:
    """
    加载YAML配置文件并转换为字典
    :param yaml_path: YAML文件路径
    :return: 包含配置参数的字典
    :raises FileNotFoundError: 当文件不存在时
    :raises yaml.YAMLError: 当YAML格式错误时
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config or {}  # 确保空文件返回空字典
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {yaml_path} not found!")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML syntax error in {yaml_path}: {str(e)}")

def get_next_exp_dir(base_dir='run') -> str:
    """
    在 base_dir 下创建下一个 expN 文件夹并返回路径
    """
    os.makedirs(base_dir, exist_ok=True)
    # 查找已有 expN
    max_idx = -1
    for name in os.listdir(base_dir):
        if name.startswith('exp'):
            try:
                idx = int(name[3:])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    next_idx = max_idx + 1
    exp_path = os.path.join(base_dir, f"exp{next_idx}")
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def main():
    exp_dir = get_next_exp_dir('run')
    print(f"Using experiment dir: {exp_dir}")
    # Load configuration and initialize CVAutoFuzz
    config = load_yaml("config.yaml")
    autofuzz = CVAutoFuzz(config)
    # fuzzer = CANFuzzer()
    os.chdir(exp_dir)

    # 获取配置中的 CAN 参数，如果没有则使用默认值
    can_config = config.get('can', {})
    can_channel = can_config.get('channel', 'can0')
    can_bustype = can_config.get('bustype', 'socketcan')
    can_id_start = can_config.get('id_start', 0x100)
    can_id_end = can_config.get('id_end', 0x1FF)
    can_send_delay = can_config.get('send_delay', 0.01)
    # 实例化时传入参数
    fuzzer = CANFuzzer(channel=can_channel, bustype=can_bustype,
                       id_start=can_id_start, id_end=can_id_end,send_delay=can_send_delay)
    print(
        f"Initialized CANFuzzer on channel: {can_channel}, bustype: {can_bustype}, id_range=0x{int(can_id_start):X}-0x{int(can_id_end):X}, send_delay={can_send_delay}s")

    # 从配置读取总测试时长（小时），默认 1 小时
    total_hours = float(config.get('test_hours', 1))
    print(f"Total test duration set to {total_hours} hours.")
    end_time = time.time() + total_hours * 3600
    round_idx = 1
    while True:
        round_start = time.time()
        print(f"\n=== Starting test round {round_idx} ===")


        # Start the vision detection in a separate thread
        result_queue = Queue()
        # 确保上一轮遗留文件不会累积（可选）
        if os.path.exists("can_temp.txt"):
            try:
                os.remove("can_temp.txt")
            except Exception:
                pass

        autofuzz.start_detection(result_queue)
        print("vision detector need 5 second to started")
        # time.sleep(5)
        # Launch the CAN fuzzing to send 1000 messages (can_temp.txt will be generated)
        # 本地测试时需要先注释,不然会在发送CAN报文时报错(前置条件:CAN通道已经建立)

        for i in range(5):
             time.sleep(1)
             print(i)
        # # time.sleep(15)
        # 执行一轮模糊测试（会生成 can_temp.txt 并发送）
        fuzzer.start_fuzzing()

        # Stop the detection thread after fuzzing is done and collect error labels
        autofuzz.stop_detection()
        error_list = []
        if not result_queue.empty():
            error_list = result_queue.get()  # Retrieve the list of error labels detected

        # If any error labels were detected during the initial round, perform binary search on messages
        if error_list:
            # 1. 去重：获取所有唯一的错误类型
            unique_errors = list(set(error_list))
            print(f"Detected unique errors: {unique_errors}")

            # 2. 针对每一个错误，单独运行一轮二分查找
            for target_error in unique_errors:
                print(f"\n{'=' * 20}")
                print(f"Starting Bisect for specific error: {target_error}")
                print(f"{'=' * 20}")

                # 每次针对新错误时，重置输入文件为完整的日志
                input_file = "can_temp.txt"

                # 这里的逻辑是为了防止找不到文件，或者文件已被覆盖
                # 实际应用中建议把 can_temp.txt 复制一份作为 base

                while True:
                    # 读取当前文件行数
                    with open(input_file, 'r') as f:
                        lines = [line for line in f.read().splitlines() if line.strip()]

                    if len(lines) < 8:
                        print(f"Bisect finished for {target_error}. Messages reduced to {len(lines)}.")
                        break

                    # 切分文件
                    fileA, fileB = autofuzz.split_file(input_file)

                    # 3. 关键修改：传入 target_label
                    # 先测 A 半区
                    print(f"Testing split A for {target_error}...")
                    if autofuzz.send_and_detect(fileA, target_label=target_error):
                        input_file = fileA
                        print("  -> Error found in A. Discarding B.")
                        continue  # 直接进入下一次循环

                    # 如果 A 没触发，测 B 半区
                    print(f"Testing split B for {target_error}...")
                    if autofuzz.send_and_detect(fileB, target_label=target_error):
                        input_file = fileB
                        print("  -> Error found in B. Discarding A.")
                        continue

                    # 如果两边都没触发 (可能因为错误是间歇性的，或者被切分打断了时序)
                    print(f"Error {target_error} not reproduced in either split. Stopping bisect.")
                    break

                # 4. 为当前这个错误生成独立的报告
                # input_file 此时是针对该错误的最小触发集
                autofuzz.generate_error_report([target_error], input_file)

        else:
            print("No errors were detected in the first round of fuzzing.")
        # 轮次结束后检查是否达到总测试时间；若已到则退出，否则开始下一轮
        now = time.time()
        if now >= end_time:
            print(f"Total test time reached after round {round_idx}. Exiting.")
            break
        else:
            remaining = end_time - now
            print(f"Round {round_idx} finished. Remaining test time: {remaining/3600:.2f} hours.")
            round_idx += 1
            # 继续下一轮（检测线程和 fuzzer 状态在 start/stop 中已处理）

def adaptive_main():
    # 1. 读配置
    config = load_yaml("config.yaml")
    vision_cfg = config.get("vision", {})
    can_cfg = config.get("can", {})

    # 2. 初始化 VisionDetector 和 CANFuzzer
    detector = VisionDetector(**vision_cfg)
    can_channel = can_cfg.get('channel', 'can0')
    can_bustype = can_cfg.get('bustype', 'socketcan')
    can_id_start = can_cfg.get('id_start', 0x180)
    can_id_end = can_cfg.get('id_end', 0x189)
    can_send_delay = can_cfg.get('send_delay', 0.01)

    can_fuzzer = CANFuzzer(
        channel=can_channel,
        bustype=can_bustype,
        id_start=can_id_start,
        id_end=can_id_end,
        send_delay=can_send_delay
    )

    print(
        f"[AdaptiveMain] CANFuzzer on {can_channel}/{can_bustype}, "
        f"ID range=0x{int(can_id_start):X}-0x{int(can_id_end):X}"
    )

    # 3. 初始化自适应 fuzzer（可将参数写进 config.yaml）
    adaptive = VisionGuidedAdaptiveFuzzer(
        can_fuzzer=can_fuzzer,
        detector=detector,
        id_start=can_id_start,
        id_end=can_id_end,
        # 如果有 sniff 得到的 seed IDs，可以传进来，
        # 没有的话可以不填，使用 [id_start, id_end] 全范围
        # seed_ids=[0x180, 0x181, 0x182],
        epsilon=0.4,
        alpha=1.0,
        beta=1.0,
        default_freq_hz=10.0,
        frames_per_episode=20,
        settle_time=0.1, #发送完一组报文之后的等待时间，应该大于灯熄灭时间
        min_byte_trials_for_bit=2,
        byte_reward_threshold_for_bit=1.0,
        neighbor_delta=0x1,
        neighbor_reward_threshold=2.0,
        neighbor_min_trials=10,
        log_dir="logs1",
        min_bit_events_for_mapping=1,
        min_confidence_for_mapping=0.4,
    )

    # 4. 运行若干 episode（例如 500 个）
    try:
        adaptive.run(num_episodes=5000)
    finally:
        detector.release()
        can_fuzzer.bus.shutdown()

def super_adaptive_main():
    # 1. 读配置
    config = load_yaml("config.yaml")
    vision_cfg = config.get("vision", {})
    can_cfg = config.get("can", {})

    # 2. 初始化 VisionDetector 和 CANFuzzer
    detector = VisionDetector(**vision_cfg)
    can_channel = can_cfg.get('channel', 'can0')
    can_bustype = can_cfg.get('bustype', 'socketcan')
    can_id_start = can_cfg.get('id_start', 0x180)
    can_id_end = can_cfg.get('id_end', 0x189)
    can_send_delay = can_cfg.get('send_delay', 0.01)

    can_fuzzer = CANFuzzer(
        channel=can_channel,
        bustype=can_bustype,
        id_start=can_id_start,
        id_end=can_id_end,
        send_delay=can_send_delay
    )

    print(
        f"[AdaptiveMain] CANFuzzer on {can_channel}/{can_bustype}, "
        f"ID range=0x{int(can_id_start):X}-0x{int(can_id_end):X}"
    )
    fuzzer = SuperVisionGuidedAdaptiveFuzzer(
        can_fuzzer=can_fuzzer,
        detector=detector,
        id_start=0x100,
        id_end=0x1FF,
        epsilon=0.2,
        alpha=1.0,
        beta=5.0,
        default_freq_hz=20.0,
        baseline_repeats=1,
        mutated_repeats=3,  # 有利于触发类似 “2000ms 内 3 条报文” 条件
        settle_time=0.1,
        lamp_reset_time=0.6,
        global_min_trials_per_id=5,
        max_trials_per_id=500,
        neighbor_delta=1,
        neighbor_min_trials=10,
        neighbor_reward_threshold=1.0,
        multi_combo_period=50,  # 每 50 个 episode 做一次多 ID 组合尝试
        min_events_for_combo=3,
        log_dir="logsSuper",
        min_bit_events_for_mapping=5,
        min_confidence_for_mapping=0.6,
    )

    fuzzer.run(num_episodes=5000)



if __name__ == "__main__":
    super_adaptive_main()
# 调试阶段先调用 adaptive_main()
#     adaptive_main()
    # 或根据需要保留原 main()
    # main()