
import os
import time
import random
from datetime import datetime

import cv2

from CAN_fuzzer import send_can_messages_from_file

SAVE_DIR = "data_set"
CAN_BATCH_DIR = "can_batches"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CAN_BATCH_DIR, exist_ok=True)

LIGHT_IDS = [0x180 + i for i in range(10)]
FRAMES_TARGET = 200
MAX_COMBO = 3
LIGHT_ON_DURATION = 0.5  # seconds

CAM_INDEX = 0
RESOLUTION = (640, 480)

CAN_CHANNEL = "can0"
CAN_INTERFACE = "socketcan"
SEND_DELAY = 0.01  # 与 send_can_messages_from_file 保持一致


def generate_plan(num_frames: int):
    return [
        sorted(random.sample(LIGHT_IDS, random.randint(1, MAX_COMBO)))
        for _ in range(num_frames)
    ]


def prepare_can_batches(plan):
    batch_files = []
    aggregate_path = os.path.join(CAN_BATCH_DIR, "all_batches.txt")
    with open(aggregate_path, "w") as aggregate:
        for idx, combo in enumerate(plan):
            file_path = os.path.join(CAN_BATCH_DIR, f"batch_{idx:04d}.txt")
            with open(file_path, "w") as f:
                for arb_id in combo:
                    line = f"{arb_id:03X}#01"
                    f.write(line + "\n")
                    aggregate.write(line + "\n")
            batch_files.append(file_path)
    return batch_files


def capture_dataset():
    plan = generate_plan(FRAMES_TARGET)
    batch_files = prepare_can_batches(plan)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败")

    saved = 0
    try:
        for batch_file in batch_files:
            send_can_messages_from_file(
                batch_file,
                channel=CAN_CHANNEL,
                interface=CAN_INTERFACE,
                send_delay=SEND_DELAY,
            )

            time.sleep(LIGHT_ON_DURATION)

            ret, frame = cap.read()
            if not ret:
                print("读取帧失败，跳过该批次")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"yolov5lite_{saved:04d}_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            saved += 1
            print(f"[{saved}/{FRAMES_TARGET}] 保存 {filepath}")

            if saved >= FRAMES_TARGET:
                break
    finally:
        cap.release()


if __name__ == "__main__":
    random.seed()
    capture_dataset()
