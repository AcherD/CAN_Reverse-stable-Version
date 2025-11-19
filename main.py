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
    fuzzer = CANFuzzer()

    # 获取配置中的 CAN 参数，如果没有则使用默认值
    can_config = config.get('can', {})
    can_channel = can_config.get('channel', 'can0')
    can_bustype = can_config.get('bustype', 'socketcan')
    can_id_start = can_config.get('id_start', 0x100)
    can_id_end = can_config.get('id_end', 0x1FF)
    # 实例化时传入参数
    fuzzer = CANFuzzer(channel=can_channel, bustype=can_bustype,
                       id_start=can_id_start, id_end=can_id_end)
    print(
        f"Initialized CANFuzzer on channel: {can_channel}, bustype: {can_bustype}, id_range=0x{int(can_id_start):X}-0x{int(can_id_end):X}")


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
        input_file = "can_temp.txt"
        # Narrow down the set of messages that trigger the error
        while True:
            # Read all CAN message lines from the current file
            with open(input_file, 'r') as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            if len(lines) < 10:
                # Stop narrowing down when fewer than 10 messages remain
                break
            # Split the current file into two halves (A.txt and B.txt)
            fileA, fileB = autofuzz.split_file(input_file)
            # Test each half to see which still triggers the error
            resultA = autofuzz.send_and_detect(fileA)
            resultB = False
            if resultA:
                # Error is triggered by messages in A.txt; continue with A.txt
                input_file = fileA
            else:
                # If A half did not trigger, test B half
                resultB = autofuzz.send_and_detect(fileB)
                if resultB:
                    # Error is triggered by messages in B.txt; continue with B.txt
                    input_file = fileB
                else:
                    # Neither half triggered an error (unexpected if initial had error)
                    print("Error not observed in either split; cannot isolate further.")
                    break

        final_file = input_file  # The narrowed-down file with <10 messages
        # Generate a comprehensive error report including the error labels and final messages
        autofuzz.generate_error_report(error_list, final_file)
    else:
        print("No errors were detected in the first round of fuzzing.")


if __name__ == "__main__":
    main()
