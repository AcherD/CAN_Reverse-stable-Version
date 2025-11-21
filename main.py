import threading
import time
from multiprocessing import Process, Manager
from VisionDetector import VisionDetector
from CAN_fuzzer import CANFuzzer
from CVAutoFuzz import CVAutoFuzz
import yaml
from pathlib import Path
from queue import Queue


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

def main():
    # Load configuration and initialize CVAutoFuzz
    config = load_yaml("config.yaml")
    autofuzz = CVAutoFuzz(config)
    # fuzzer = CANFuzzer()

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


    # Start the vision detection in a separate thread
    result_queue = Queue()
    autofuzz.start_detection(result_queue)
    print("vision detector need 5 second to started")
    # time.sleep(5)
    # Launch the CAN fuzzing to send 1000 messages (can_temp.txt will be generated)
    # 本地测试时需要先注释,不然会在发送CAN报文时报错(前置条件:CAN通道已经建立)

    for i in range(5):
         time.sleep(1)
         print(i)
    # # time.sleep(15)
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

                if len(lines) < 10:
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


if __name__ == "__main__":
    main()
