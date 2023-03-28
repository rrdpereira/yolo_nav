import pyrealsense2 as rs
import numpy as np
import cv2
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline = rs.pipeline()
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(f"(intr) : {intr}")
print(f"(intr.fx, intr.fy) : {intr.fx}, {intr.fy}")

x = 320
y = 240
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    depth_frame_ = np.asanyarray(depth_frame.get_data())

    depth_frame_ = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame_, alpha=0.03), cv2.COLORMAP_JET)
    color_image = np.asanyarray(color_frame.get_data())
    dist = depth_frame.get_distance(int(x), int(y))
    pos = rs.rs2_deproject_pixel_to_point(intr, [x, y], dist)
    Xtemp = ((dist * (x - intr.ppx) / intr.fx) * 1000)
    Ytemp = (dist * (y - intr.ppy) / intr.fy) * 1000
    Ytemp = Ytemp
    Ztemp = (dist * 1000)
    Ztemp = Ztemp
    cv2.circle(color_image, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imshow("k", color_image)

    if cv2.waitKey(100) == ord('q'):
        break

    print("Own_Calculation:", Ytemp)
    print("Realsense: ", pos[1]*1000)