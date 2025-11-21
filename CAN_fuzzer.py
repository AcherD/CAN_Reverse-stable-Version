import random
from caringcaribou.modules import fuzzer,send
import can
import time
import os
DEFAULT_SEED_MAX = 2 ** 16
#二分法
class Bisecter:
    def __init__(self, candidate_msgs):
        self.candidate = candidate_msgs

    def find_trigger(self, test_func):
        low = 0
        high = len(self.candidate) - 1

        while low < high:
            mid = (low + high) // 2
            if test_func(self.candidate[:mid + 1]):
                high = mid
            else:
                low = mid + 1
        return self.candidate[low]

class CANFuzzer:
    def __init__(self, channel='can0', bustype='socketcan', max_freq=100, id_start=0x100, id_end=0x1FF,send_delay=0.01):
        self.channel = channel
        self.bustype = bustype
        self.bus = can.interface.Bus(channel=channel, bustype=bustype)
        self.sent_log = []
        self.min_interval = 1.0 / max_freq  # 每秒最多发送max_freq帧
        self.last_send_time = time.time()
        # ID 范围
        self.id_start = int(id_start)
        self.id_end = int(id_end)
        # 发送控制标志
        self.recording = False
        # 从配置传入的发送延迟（用于文件重放）
        self.send_delay = send_delay

    def generate_random_msg(self):
        # 使用实例范围生成随机CAN ID和数据
        arb_id = random.randint(self.id_start, self.id_end)
        data = bytes([random.randint(0, 255) for _ in range(8)])
        return can.Message(arbitration_id=arb_id, data=data)

    def send_msg(self, msg):
        try:
            # 添加速率限制
            elapsed = time.time() - self.last_send_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)

            self.bus.send(msg)
            self.last_send_time = time.time()
            self.sent_log.append((self.last_send_time, msg))
        except can.CanError as e:
            print(f"发送失败: {str(e)}")
            # 可选：重试逻辑
            # self._retry_send(msg)


    def start_fuzzing(self, duration=120):
        start_time = time.time()
        # 生成1000条随机数据，进行存储
        filename = "can_temp.txt"
        output_file = open(filename, "a")
        # 生成种子
        set_seed()
        for i in range(0, 1000):
            # 使用实例方法生成
            msg = self.generate_random_msg()
            arb_id = msg.arbitration_id
            data = msg.data
            write_directive_to_file_handle(output_file, arb_id=arb_id, data=data)
        output_file.close()
        # 发送生成的所有报文
        send_can_messages_from_file("can_temp.txt", channel=self.channel, interface=self.bustype,send_delay=self.send_delay)

        # while time.time() - start_time < duration:
        #     msg = self.generate_random_msg()
        #     self.send_msg(msg)
        #     time.sleep(0.01)  # 控制发送频率

    def save_context(self, trigger_time, before=3, after=3):
        # 提取触发时刻前后3秒的报文
        return [entry for entry in self.sent_log
                if trigger_time - before < entry[0] < trigger_time + after]

    def bisect_trigger(self, candidate_msgs):
        # 使用二分法定位触发报文
        return Bisecter(candidate_msgs).find_trigger(self._test_case)

    def _test_case(self, msgs):
        # 重放报文并检测错误
        for _, msg in msgs:
            self.bus.send(msg)
            time.sleep(0.01)  # 保持时序
        return self._check_error_occurred()  # 需要与视觉检测联动

    def send_random_message(self):
        fuzzer.random_fuzz(filename="fuzzed_data.txt")

    def clean(self):
        if os.path.exists("fuzzed_data.txt"):
            os.remove("fuzzed_data.txt")

# 以下为工具类

# 从文件中发送报文
def send_can_messages_from_file(file_path, channel='can0', interface='socketcan', bitrate=500000, send_delay=0.01):
    """
    发送CAN报文
    参数：
        file_path: 包含CAN数据的文件路径
        channel: CAN接口通道（默认：can1）
        interface: CAN接口类型（默认：socketcan）
        bitrate: 波特率（默认：500000）
    """
    try:
        # 配置CAN总线
        bus = can.Bus(
            channel=channel,
            interface=interface,
            bitrate=bitrate
        )

        # 读取文件内容
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 解析并发送每条报文
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 分割ID和数据
            try:
                can_id, data = line.split('#')
                can_id = int(can_id, 16)  # 将十六进制字符串转换为整数
                data_bytes = bytes.fromhex(data)  # 将十六进制字符串转换为字节
            except ValueError as e:
                print(f"格式错误：{line} -> {e}")
                continue

            # 创建CAN消息（假设使用扩展帧）
            msg = can.Message(
                arbitration_id=can_id,
                data=data_bytes,
                is_extended_id=True  # 根据ID长度自动判断更安全
            )

            # 发送消息
            bus.send(msg)
            print(f"已发送：ID=0x{can_id:X}, Data={data_bytes.hex()}")
            time.sleep(send_delay)  # 避免总线拥塞

        bus.shutdown()
    except Exception as e:
        print(f"发送失败：{e}")


# 设置种子
def set_seed(seed=None):
    """
    Seeds the PRNG with 'seed'. If this is None, a seed is pulled from the PRNG instead.

    :param seed: int to use for seeding
    """
    if seed is None:
        seed = random.randint(0, DEFAULT_SEED_MAX)
    print("Seed: {0} (0x{0:x})".format(seed))
    random.seed(seed)


# 模块级兼容函数，允许调用时指定范围
def generate_random_msg(id_start=0x100, id_end=0x1FF):
    arb_id = random.randint(int(id_start), int(id_end))
    data = bytes([random.randint(0, 255) for _ in range(8)])
    return can.Message(arbitration_id=arb_id, data=data)


def list_to_hex_str(data, delimiter=""):
    """Returns a hex string representation of the int values
    in 'data', separated with 'delimiter' between each byte

    Example:
    list_to_hex_str([10, 100, 200]) -> 0a.64.c8
    list_to_hex_str([0x07, 0xff, 0x6c], "") -> 07ff6c
    :param data: iterable of values
    :param delimiter: separator between values in output
    :type data: [int]
    :type delimiter: str
    :return: hex string representation of data
    :rtype str
    """
    data_string = delimiter.join(["{0:02x}".format(i) for i in data])
    return data_string

def directive_str(arb_id, data):
    """
    Converts a directive to its string representation

    :param arb_id: message arbitration ID
    :param data: message data bytes
    :return: str representing directive
    """
    data = list_to_hex_str(data, "")
    directive = "{0:03X}#{1}".format(arb_id, data)
    return directive


def write_directive_to_file_handle(file_handle, arb_id, data):
    """
    Writes a cansend directive to a file

    :param file_handle: handle for the output file
    :param arb_id: int arbitration ID
    :param data: list of data bytes
    """
    directive = directive_str(arb_id, data)
    file_handle.write("{0}\n".format(directive))