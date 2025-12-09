# python
import os
import cv2
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import base64
from io import BytesIO

# 尝试导入 PIL 的 Image 与 ImageTk；如果 ImageTk 不可用，启用回退路径
try:
    from PIL import Image, ImageTk
    _HAS_IMAGETK = True
except Exception:
    try:
        from PIL import Image
        _HAS_IMAGETK = False
    except Exception:
        raise RuntimeError(
            "Pillow 不可用。请安装 Pillow 或运行：\n"
            "  sudo apt install python3-tk\n"
            "  python3 -m pip install --upgrade Pillow"
        )

SAVE_DIR = "data_set"
os.makedirs(SAVE_DIR, exist_ok=True)

class CameraApp:
    def __init__(self, root, cam_index=0, width=640, height=480):
        self.root = root
        self.root.title("摄像头拍照工具")
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            root.destroy()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=6)

        self.capture_btn = tk.Button(btn_frame, text="拍照", width=12, command=self.capture)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.quit_btn = tk.Button(btn_frame, text="退出", width=12, command=self.on_closing)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        self.current_frame = None
        self._photo_ref = None  # 保存 PhotoImage 引用防止被回收
        self.update_frame()
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _frame_to_tk(self, frame_bgr):
        # 转为 RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if _HAS_IMAGETK:
            # 使用 PIL.ImageTk（优先）
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            return img_tk
        else:
            # 回退：将 RGB 图像编码为 PNG，然后 base64 编码，交给 tk.PhotoImage
            success, buf = cv2.imencode('.png', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if not success:
                return None
            b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            try:
                img_tk = tk.PhotoImage(data=b64)
            except Exception:
                # 某些 tk 版本需要 'data' 为 base64 字符串前加 header，尝试带 header
                img_tk = tk.PhotoImage(data="base64," + b64)
            return img_tk

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.video_label.config(text="无法读取摄像头视频流")
            self.root.after(1000, self.update_frame)
            return

        self.current_frame = frame.copy()
        img_tk = self._frame_to_tk(self.current_frame)
        if img_tk is None:
            self.video_label.config(text="无法将帧转换为显示格式")
            self.root.after(1000, self.update_frame)
            return

        self._photo_ref = img_tk
        self.video_label.config(image=img_tk)
        self.root.after(30, self.update_frame)

    def capture(self):
        if self.current_frame is None:
            messagebox.showwarning("警告", "当前没有可用帧")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, self.current_frame)
        messagebox.showinfo("已保存", f"图片已保存到：{filepath}")

    def on_closing(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
