import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
stream_profile_color = profile.get_stream(rs.stream.color)
stream_profile_depth = profile.get_stream(rs.stream.depth)
color_intrs = stream_profile_color.as_video_stream_profile().get_intrinsics()
depth_intrs = stream_profile_depth.as_video_stream_profile().get_intrinsics()
extrinsics = stream_profile_depth.get_extrinsics_to(stream_profile_color)
print("color_intrs:\n", color_intrs)
print("depth_intrs:\n", depth_intrs)
print("extrins:\n", extrinsics)
print(f"(color_intrs.fx, color_intrs.fy,color_intrs.ppx, color_intrs.ppy) : {color_intrs.fx}, {color_intrs.fy},{color_intrs.ppx}, {color_intrs.ppy}")
print(f"(depth_intrs.fx, depth_intrs.fy,depth_intrs.ppx, depth_intrs.ppy) : {depth_intrs.fx}, {depth_intrs.fy},{depth_intrs.ppx}, {depth_intrs.ppy}")
align_to = rs.stream.color
align = rs.align(align_to)

# pixel coordinate
x = 320
y = 240
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
aligned_depth_frames = aligned_frames.get_depth_frame()
dis = aligned_depth_frames.get_distance(x, y)

# get the 3D coordinate 
camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrs, [x, y], dis)
camera_coordinate1 = rs.rs2_deproject_pixel_to_point(color_intrs, [x, y], dis)

print("distance:", dis)
print("depth intrins:", camera_coordinate)
print("color intrins:", camera_coordinate1)
pipeline.stop()