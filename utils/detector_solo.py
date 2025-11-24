# python
import cv2
import numpy as np
import onnxruntime as ort

# 模型与类别（与训练时顺序一致）
MODEL_PATH = 'best.onnx'
labels = [
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
]

# 锚点与阈值（替换为与你模型一致的锚点）
anchors = [
    [[10, 13], [16, 30], [33, 23]],   # stride 8
    [[30, 61], [62, 45], [59, 119]],  # stride 16
    [[116, 90], [156, 198], [373, 326]]  # stride 32
]
conf_threshold = 0.7
iou_threshold = 0.5

# 加载 ONNX 模型
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
# 兼容动态输入形状（默认 640）
inp_shape = session.get_inputs()[0].shape
model_h = int(inp_shape[2]) if len(inp_shape) > 2 and inp_shape[2] is not None else 640
model_w = int(inp_shape[3]) if len(inp_shape) > 3 and inp_shape[3] is not None else 640

def preprocess(img):
    h, w = img.shape[:2]
    scale = min(model_h / h, model_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    padded = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    top = (model_h - new_h) // 2
    left = (model_w - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized
    tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None, ...]
    return tensor, scale, (left, top)

def decode_predictions(outputs, orig_shape, anchors):
    orig_h, orig_w = orig_shape
    detections = []
    for i, output in enumerate(outputs):
        # 期望输出形状 (1,3,H,W,5+num_classes) 或 (1,3,H,W,C)
        out = np.array(output)
        if out.ndim == 4:
            # 有时每层为 (1, H, W, C) 拆分为 3 anchors: reshape 到 (1,3,H,W,C/3) 不常见，这里假设原始导出为 (1,3,H,W,C)
            out = out.reshape((1,)+out.shape[1:-1]+(out.shape[-1],))
        out = out.squeeze(0)  # [3, H, W, C]
        _, grid_h, grid_w, _ = out.shape

        stride = 8 * (2 ** i)
        grid_x = (np.arange(grid_w) + 0.5)
        grid_y = (np.arange(grid_h) + 0.5)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_xy = np.stack((gx, gy), axis=-1)[None, ...]  # [1, H, W, 2]
        grid_xy = np.repeat(grid_xy, 3, axis=0)  # [3, H, W, 2]

        anchor_array = np.array(anchors[i], dtype=np.float32).reshape(3, 1, 1, 2)

        xy = (out[..., 0:2] * 2 - 0.5 + grid_xy) * stride
        wh = (out[..., 2:4] * 2) ** 2 * anchor_array

        # 恢复到原始图像尺度
        xy[..., 0] *= orig_w / model_w
        xy[..., 1] *= orig_h / model_h
        wh[..., 0] *= orig_w / model_w
        wh[..., 1] *= orig_h / model_h

        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        boxes = np.concatenate((x1y1, x2y2), axis=-1)  # [3,H,W,4]

        conf = out[..., 4]
        class_ids = np.argmax(out[..., 5:], axis=-1)
        mask = conf > conf_threshold

        for a in range(3):
            idx = mask[a]
            if np.any(idx):
                bx = boxes[a][idx]
                bf = conf[a][idx]
                bc = class_ids[a][idx]
                for j in range(bx.shape[0]):
                    detections.append([bx[j,0], bx[j,1], bx[j,2], bx[j,3], float(bf[j]), int(bc[j])])
    if len(detections) == 0:
        return np.zeros((0,6))
    return np.array(detections)

def nms_and_draw(frame, detections):
    if detections.shape[0] == 0:
        return frame
    boxes_xyxy = detections[:, :4]
    scores = detections[:, 4]
    class_ids = detections[:, 5].astype(int)

    # 转换为 x,y,w,h 用于 cv2.dnn.NMSBoxes
    boxes_xywh = []
    for b in boxes_xyxy:
        x1, y1, x2, y2 = b
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0:
        return frame
    # 处理不同返回格式
    if isinstance(indices, (np.ndarray, list)):
        inds = np.array(indices).flatten()
    else:
        inds = np.array([indices]).flatten()

    for i in inds:
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cls_id = class_ids[i]
        label = f"{labels[cls_id]} {scores[i]:.2f}"
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            print(f"识别到 [{labels[cls_id]}] {scores[i]:.2f}")
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_tensor, scale, (left_pad, top_pad) = preprocess(frame)
            # 推理
            outputs = session.run(None, {input_name: input_tensor.astype(np.float32)})
            detections = decode_predictions(outputs, frame.shape[:2], anchors)
            frame = nms_and_draw(frame, detections)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
