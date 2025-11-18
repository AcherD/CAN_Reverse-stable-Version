import os

from caringcaribou.modules import fuzzer,send
import can



class fuzzer_caring:
    def __init__(self, channel='vcan0', bustype='socketcan'):
        self.bus = can.interface.Bus(channel=channel, bustype=bustype)
        self.sent_log = []
        self.recording = False

    def send_random_message(self):
        fuzzer.random_fuzz(filename="fuzzed_data.txt")

    def clean(self):
        if os.path.exists("fuzzed_data.txt"):
            os.remove("fuzzed_data.txt")

    def replay(self,replay_file='fuzzed_data.txt',delay=0.2):
        messages = send.parse_file(replay_file,delay)
        send.send_messages(messages)

