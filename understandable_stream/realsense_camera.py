# https://pysource.com
import pyrealsense2 as rs
import numpy as np


class RealsenseCamera:
    def __init__(self, x_color, y_color, x_depth, y_depth):
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")

        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, x_color, y_color, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, x_depth, y_depth, rs.format.z16, 30)
        print(f"Configuration loaded. Start streaming with resolutions: color {x_color}x{y_color}, depth {x_depth}x{y_depth}")

        self.color_frame = None
        self.depth = None

        # Start streaming
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

    def get_frame_stream(self):
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth = aligned_frames.get_depth_frame()

        color_frame = aligned_frames.get_color_frame()

        self.color_frame = color_frame
        self.depth = depth
        
        if not depth or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None
        
        # Apply filter to fill the holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)
        
        # Create colormap to show the depth of the Objects
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return True, color_image, depth_image


    def get_distance_point(self, depth_frame, x, y):
        # Get the distance of a point in the image
        distance = self.depth.get_distance(x, y)
        # convert to cm
        return round(distance * 100, 2)

    
    def stop(self):
        self.pipeline.stop()
        print("Camera stopped")

        
