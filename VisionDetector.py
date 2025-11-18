import cv2
import onnxruntime as ort
import numpy as np


class VisionDetector:
    def __init__(self, model_path, conf_thresh=0.7, cam_id=0, debug=True):
        self.session = ort.InferenceSession(model_path)
        # 从模型元数据中获取输入尺寸
        model_input = self.session.get_inputs()[0]
        self.input_shape = model_input.shape  # 格式：[batch, channel, height, width]
        self.input_size = (self.input_shape[3], self.input_shape[2])  # (width, height)

        # 验证输入通道顺序
        if self.input_shape[1] != 3:
            raise ValueError(f"模型需要3通道输入，当前配置为{self.input_shape[1]}通道")

        # 初始化其他参数
        self.conf_thresh = conf_thresh
        self.cam = cv2.VideoCapture(cam_id)
        self.debug = debug  # 新增：调试模式开关
        self.labels = ['Anti Lock Braking System',
                       'Braking System Issue',
                       'Charging System Issue',
                       'Check Engine',
                       'Electronic Stability Problem -ESP-',
                       'Engine Overheating Warning Light',
                       'Low Engine Oil Warning Light',
                       'Low Tire Pressure Warning Light',
                       'Master warning light',
                       'SRS-Airbag']  # 确保与训练时的类别顺序一致
        self.anchors = [[10, 13],  # 对应anchor_idx=0
                        [16, 30],  # anchor_idx=1
                        [33, 23]]
        self.stride = 32  # 对应输出层的步长（需与模型匹配）
        # 验证输入输出格式
        self.input_name = self.session.get_inputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        print(f"Model expects input: {self.input_name}, Output shape: {self.output_shape}")

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return img

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detect(self):
        ret, frame = self.cam.read()
        if not ret: return []

        # 推理
        blob = self.preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})

        # 解析YOLO原始输出
        detections = []
        grid = outputs[0][0]  # 形状 [3,40,40,15]

        # 遍历每个锚点
        for anchor_idx in range(grid.shape[0]):
            anchor = self.anchors[anchor_idx]

            # 遍历特征图每个位置
            for i in range(grid.shape[2]):
                for j in range(grid.shape[1]):
                    # 提取预测数据
                    pred = grid[anchor_idx, j, i]

                    # 解析坐标和置信度
                    x, y, w, h, conf = pred[:5]
                    cls_probs = pred[5:]

                    # 应用sigmoid处理
                    conf = self._sigmoid(conf).item()  # 转换为标量
                    if conf < self.conf_thresh:
                        continue

                    # 计算实际坐标（需反算到原图尺寸）
                    cx = (self._sigmoid(x) * 2 - 0.5 + j) * self.stride
                    cy = (self._sigmoid(y) * 2 - 0.5 + i) * self.stride
                    width = (self._sigmoid(w) * 2) ** 2 * anchor[0]
                    height = (self._sigmoid(h) * 2) ** 2 * anchor[1]

                    # 获取类别
                    cls_id = np.argmax(cls_probs)

                    detections.append({
                        "label": self.labels[cls_id],
                        "conf": conf,
                        "bbox": [cx, cy, width, height],
                        "frame": frame.copy()
                    })

        if self.debug:
            # 在当前帧上绘制所有检测结果
            for det in detections:
                label = det["label"]
                conf_val = det["conf"]
                cx, cy, w, h = det["bbox"]
                # 计算矩形框的顶点坐标（转换为整数）
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
                # 绘制边界框和标签文字
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf_val:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 显示调试窗口并处理退出按键
            cv2.imshow("Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.debug = False  # 关闭调试模式（退出调试显示）

        return detections

    def release(self):
        self.cam.release()
