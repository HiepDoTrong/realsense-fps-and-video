from PIL import Image
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# Tạo thư mục để lưu ảnh
output_folder = "/home/dotronghiep/Documents/Research/Peanuts_Anomaly_Detection_PAD/realsense-fps-and-video"
os.makedirs(output_folder, exist_ok=True)

# Khởi tạo pipeline và cấu hình
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Bắt đầu pipeline
pipe.start(cfg)
count = 0

try:
    while True:
        # Chờ cho đến khi có dữ liệu hình ảnh
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Chuyển đổi sang mảng NumPy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Hiển thị và lưu ảnh
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_image)

        # Lưu ảnh vào thư mục
        color_filename = os.path.join(output_folder, "color_images", f"color_{count}.png")
        depth_filename = os.path.join(output_folder, "depth_images", f"depth_{count}.tiff")

        # Lưu ảnh chiều sâu dưới định dạng TIFF
        depth_image_pil = Image.fromarray(depth_image)
        depth_image_pil.save(depth_filename, format="TIFF")

        cv2.imwrite(color_filename, color_image)

        # Chờ 1 giây
        time.sleep(1)
        count+=1

        # Nếu người dùng nhấn phím ESC, thoát vòng lặp
        if cv2.waitKey(1) == 27:
            break

finally:
    # Dừng pipeline khi kết thúc
    pipe.stop()
    cv2.destroyAllWindows()
