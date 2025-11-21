import cv2
import threading
from reportlab.pdfgen import canvas
from datetime import datetime, time
from VisionDetector import VisionDetector
from CAN_fuzzer import CANFuzzer
import time
from queue import Queue


class CVAutoFuzz:
    def __init__(self, config):
        self.threads_running = True
        self.detector = VisionDetector(**config['vision'])
        self.can_config = config.get('can',{})
        # self.fuzzer = CANFuzzer(**config['can'])
        self.error_labels = config['error_labels']  # 需要触发二分法的错误标签

# 先启动识别线程
# 伪代码
#     while flag：
#         启动识别线程
#         启动报文发送线程
#         报文发送结束后将flag置为false
#     结束识别线程（先试一下置为false之后会不会直接停，直接停最方便）已实现

    # def detector_run(self, queue=None):
    #     print("detection running...")
    #     error_list = []
    #     while self.threads_running:
    #         time.sleep(1)
    #         detections = self.detector.detect()
    #         for d in detections:
    #             if d['label'] in self.error_labels:
    #                 # self._handle_error(d)
    #                 error_list.append(d['label'])
    #                 print("found",d['label'])
    #     if queue is not None:
    #         queue.put(error_list)
    #
    # # 停止
    # #         if self.stop_threads:
    # #             break


    def detector_stop(self):
        self.threads_running = False
        print("detection stopped...")

    def run(self, test_duration=120):
        try:# 启动模糊测试线程
            fuzz_thread = threading.Thread(target=self.fuzzer.start_fuzzing(), args=(test_duration,))
            fuzz_thread.start()
            # 识别线程
            while True:
                detections = self.detector.detect()
                for d in detections:
                    if d['label'] in self.error_labels:
                        self._handle_error(d)
                        break

        finally:
            self.detector.release()
            self.fuzzer.bus.shutdown()

    def error_handler(self, detection):
        # 生成报告
        self._generate_report(detection)
    def _handle_error(self, detection):
        # 停止模糊测试
        self.fuzzer.recording = True  # 通过标志位停止发送线程

        # 保存上下文报文
        trigger_time = time.time()
        context = self.fuzzer.save_context(trigger_time)

        # 定位触发报文
        trigger_msg = self.fuzzer.bisect_trigger(context)

        # 生成报告
        self._generate_report(detection, trigger_msg)

    def _generate_report(self, detection, trigger_msg):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"report_{ts}.pdf"

        # 保存截图
        cv2.imwrite(f"error_{ts}.jpg", detection['frame'])

        # 生成PDF
        c = canvas.Canvas(pdf_path)
        c.drawString(100, 800, f"Error Trigger Report ({ts})")
        c.drawString(100, 750, f"Error Type: {detection['label']}")
        c.drawString(100, 700, f"CAN Message: {trigger_msg}")
        c.drawImage(f"error_{ts}.jpg", 100, 400, width=400, height=300)
        c.save()

    def start_detection(self, queue):
        """
        Start the visual detection in a separate thread.
        Detected error labels will be collected and put into the provided queue.
        """
        # Reset flags and storage for a new detection session
        self.threads_running = True
        self.error_images_map = {}
        # Launch the detector_run loop in a background thread
        self.detection_thread = threading.Thread(target=self.detector_run, args=(queue,))
        self.detection_thread.start()
        print("Visual detection thread started.")

    def stop_detection(self):
        """
        Stop the visual detection thread and wait for it to finish.
        """
        # Signal the detection thread to stop and wait for it to exit
        self.threads_running = False
        if self.detection_thread is not None:
            self.detection_thread.join()
        print("Visual detection thread stopped.")

    def detector_run(self, queue=None):
        """
        Continuous detection loop running in a thread.
        It captures frames and detects objects, collecting any error labels.
        When stopped, it puts the list of error labels into the queue.
        """
        print("Detection running...")
        error_list = []
        error_seen = set()  # track unique error labels seen
        while self.threads_running:
            time.sleep(0.01)  # throttle detection to roughly 1 frame per second
            detections = self.detector.detect()
            for d in detections:
                label = d['label']
                if label in self.error_labels and label not in error_seen:
                    # New error label detected
                    error_seen.add(label)
                    error_list.append(label)
                    print(f"Error detected: {label}")
                    # Save the frame image for report (use timestamp in filename for uniqueness)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = f"error_{label}_{ts}.jpg"
                    cv2.imwrite(img_path, d['frame'])
                    self.error_images_map[label] = img_path
        # After loop, put the collected error labels into the queue (or an empty list if none)
        if queue is not None:
            queue.put(error_list)

    def split_file(self, input_file: str) -> tuple:
        """
        Split the given CAN message file into two roughly equal halves.
        Returns the filenames of the two split files (A.txt and B.txt).
        """
        with open(input_file, 'r') as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        total = len(lines)
        mid = total // 2
        # If total is odd, put the extra line in the first half
        linesA = lines[:mid] if total % 2 == 0 else lines[:mid + 1]
        linesB = lines[mid:] if total % 2 == 0 else lines[mid + 1:]
        # Write out the split files
        with open("A.txt", 'w') as fa:
            fa.write("\n".join(linesA) + ("\n" if linesA else ""))
        with open("B.txt", 'w') as fb:
            fb.write("\n".join(linesB) + ("\n" if linesB else ""))
        print(f"Split {input_file} into A.txt ({len(linesA)} lines) and B.txt ({len(linesB)} lines).")
        return "A.txt", "B.txt"

    def send_and_detect(self, file_path: str, target_label: str = None) -> bool:
        """
        发送 CAN 报文并检测视觉反馈。
        :param file_path: 包含 CAN 报文的文件路径
        :param target_label: (新增) 如果指定了该参数，只有检测到该特定标签时才返回 True
        """
        # Set up a local queue for detection results and start detection thread
        temp_queue = Queue()
        self.start_detection(temp_queue)
        # Send all CAN messages from the file (this will block until done)
        from CAN_fuzzer import send_can_messages_from_file
        channel = self.can_config.get('channel', 'can0')
        bustype = self.can_config.get('bustype', 'socketcan')

        # 显式传递参数
        send_can_messages_from_file(file_path, channel=channel, interface=bustype)
        # send_can_messages_from_file(file_path)
        # Stop detection and collect results

        self.stop_detection()
        error_labels = []
        if not temp_queue.empty():
            error_labels = temp_queue.get()
        # If any error labels were detected during this send, return True
        has_error = (len(error_labels) > 0)
        print(f"send_and_detect: Completed sending {file_path}, error_detected={has_error}.")
        if target_label:
            # 如果指定了目标，只有包含目标才算成功
            hit = target_label in error_labels
            if hit:
                print(f"  [+] Target '{target_label}' found in this split.")
            return hit
        else:
            # 没指定目标（如第一轮Fuzz），只要有任意错误就返回True
            has_error = (len(error_labels = []) > 0)
            return has_error

    def generate_error_report(self, labels: list, file_path: str):
        """
        Generate a report (PDF) containing the error labels and the CAN messages that triggered them.
        If available, captured images of the errors will be inserted into the report.
        """
        if not labels:
            print("No errors to report.")
            return
        # Prepare a timestamped PDF filename for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"ErrorReport_{timestamp}.pdf"
        c = canvas.Canvas(pdf_filename)
        # Document title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 800, f"CVAutoFuzz Error Report - {timestamp}")
        c.setFont("Helvetica", 12)
        # List error labels
        c.drawString(50, 770, "Detected Error Labels:")
        y = 750
        for lbl in labels:
            c.drawString(70, y, f"- {lbl}")
            y -= 20
        # List the messages in the final triggering file
        c.drawString(50, y, f"Final CAN Messages (from {file_path}):")
        y -= 20
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c.drawString(70, y, line)
                y -= 15
                if y < 100:  # start a new page if space is low
                    c.showPage()
                    y = 800
                    c.setFont("Helvetica", 12)
        # Insert images for each error label if available
        if self.error_images_map:
            if y < 150:
                # Start a new page for images if current page is nearly filled
                c.showPage()
                y = 800
            c.drawString(50, y, "Captured Error Indicator Frames:")
            y -= 20
            for lbl, img_path in self.error_images_map.items():
                c.drawString(70, y, f"{lbl}:")
                y -= 10
                try:
                    # Draw the saved image (scale to fit width if needed)
                    c.drawImage(img_path, 70, y - 300, width=400, height=300)
                except Exception as e:
                    c.drawString(70, y - 15, f"[Image {img_path} could not be embedded: {e}]")
                y -= 320  # move down after image
                if y < 100:  # new page if not enough space for next image
                    c.showPage()
                    y = 800
                    c.setFont("Helvetica", 12)
        c.save()
        print(f"Error report generated: {pdf_filename}")