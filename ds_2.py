import cv2
import numpy as np
import onnxruntime as ort

# ========== 配置部分 ==========
# 必须与模型训练参数完全一致！
model_path = 'best.onnx'
anchors = [  # 示例锚点（需替换为实际值）
    [[10, 13], [16, 30], [33, 23]],  # stride 8
    [[30, 61], [62, 45], [59, 119]],  # stride 16
    [[116, 90], [156, 198], [373, 326]]  # stride 32
]
conf_threshold = 0.9
iou_threshold = 0.5
labels = [  # 必须与训练时的类别顺序一致！
    'Anti Lock Braking System',
    'Braking System Issue',
    'Charging System Issue',
    'Check Engine',
    'Electronic Stability Problem -ESP-',
    'Engine Overheating Warning Light',
    'Low Engine Oil Warning Light',
    'Low Tire Pressure Warning Light',
    'Master warning light',
    'SRS-Airbag'
] # 保持原有类别

# ========== 模型初始化 ==========
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
_, _, model_h, model_w = session.get_inputs()[0].shape  # 输入尺寸


# ========== 预处理函数 ==========
def preprocess(img):
    """ 动态适配的预处理 """
    h, w = img.shape[:2]

    # 计算缩放比例并保持长宽比
    scale = min(model_h / h, model_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # 创建填充画布 (确保居中)
    padded = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    top = (model_h - new_h) // 2
    left = (model_w - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    # 转换为模型输入格式
    input_tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(input_tensor, axis=0), scale, (left, top)


# ========== 核心解码函数 ==========
def decode_predictions(outputs, orig_shape, anchors):
    """ 精确解码YOLOv5输出 """
    orig_h, orig_w = orig_shape
    detections = []

    for i, output in enumerate(outputs):
        # 获取当前层的参数
        stride = 8 * (2 ** i)  # stride计算
        _, _, grid_h, grid_w, _ = output.shape
        output = output.squeeze(0)  # 移除批次维度 [3, H, W, 15]

        # 生成网格坐标 (匹配锚点维度)
        grid_x = np.arange(grid_w) + 0.5  # 网格中心
        grid_y = np.arange(grid_h) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_xy = np.stack((grid_x, grid_y), axis=-1)  # [H, W, 2]
        grid_xy = np.repeat(grid_xy[np.newaxis, ...], 3, axis=0)  # [3, H, W, 2]

        # 调整锚点形状 [3,1,1,2]
        anchor_array = np.array(anchors[i], dtype=np.float32).reshape(3, 1, 1, 2)

        # 解码坐标 (YOLOv5官方公式)
        xy = (output[..., 0:2] * 2 - 0.5 + grid_xy) * stride
        wh = (output[..., 2:4] * 2) ** 2 * anchor_array

        # 转换为图像绝对坐标
        xy[..., 0] *= orig_w / model_w  # x坐标
        xy[..., 1] *= orig_h / model_h  # y坐标
        wh[..., 0] *= orig_w / model_w  # 宽度
        wh[..., 1] *= orig_h / model_h  # 高度

        # 转换为xyxy格式
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        boxes = np.concatenate((x1y1, x2y2), axis=-1)  # [3, H, W, 4]

        # 提取置信度和类别
        conf = output[..., 4]
        class_ids = np.argmax(output[..., 5:], axis=-1)

        # 过滤并收集结果
        mask = conf > conf_threshold
        for a in range(3):  # 遍历每个锚点
            layer_detections = np.column_stack([
                boxes[a][mask[a]],
                conf[a][mask[a]],
                class_ids[a][mask[a]]
            ])
            if len(layer_detections) > 0:
                detections.extend(layer_detections)

    return np.array(detections)


# ========== 主循环 ==========
# ========== 主循环 ==========
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 预处理
    input_tensor, scale, (left_pad, top_pad) = preprocess(frame)

    # 模型推理
    outputs = session.run(None, {input_name: input_tensor})

    # 解码输出
    detections = decode_predictions(outputs, frame.shape[:2], anchors)

    # 应用NMS
    if len(detections) > 0:
        boxes = detections[:, :4].astype(np.float32)
        scores = detections[:, 4]
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)

        # 绘制结果
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i].astype(int)
            class_id = int(detections[i, 5])  # 获取类别ID
            label_name = labels[class_id]  # 获取标签名称

            # ========== 新增：控制台打印 ==========
            print(f"识别到 [{label_name}]")

            # 组合显示标签
            label = f"{label_name} {detections[i, 4]:.2f}"

            # 坐标边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 调试显示预处理结果
    # debug_img = (input_tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Debug View', debug_img)

    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
