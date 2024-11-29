#!/usr/bin/python
import pyrealsense2 as rs
import sys, getopt
import asyncore
import numpy as np
import pickle
import socket
import struct
import cv2


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
mc_ip_address = '172.20.10.10'
port = 1024
chunk_size = 4096
#rs.log_to_console(rs.log_severity.debug)

def getDepthAndTimestamp(pipeline, depth_filter):
    frames = pipeline.wait_for_frames()
    # take owner ship of the frame for further processing
    frames.keep()
    depth_frame = frames.get_depth_frame()
    '''
    color_frame = frames.get_color_frame()
    '''
    
    if depth_frame: 
        '''
          and color_frame:
        '''
        depth_frame = depth_filter.process(depth_frame)
        # take owner ship of the frame for further processing
        depth_frame.keep()

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        ts = frames.get_timestamp()
        return depth_frame, ts
        '''
        return depth_image, color_image, ts 
        '''
    else:
        return None, None

def openPipeline():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    cfg = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = cfg.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    cfg.enable_stream(rs.stream.depth, rs.format.z16, 30)
    '''
    cfg.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    '''
    # Start streaming
    pipeline.start(cfg)
    # sensor = pipeline_profile.get_device().first_depth_sensor()
    return pipeline


class DevNullHandler(asyncore.dispatcher_with_send):
    def handle_read(self):
        print(self.recv(1024))

    def handle_close(self):
        self.close()
           
		
class EtherSenseServer(asyncore.dispatcher):
    def __init__(self, address):
        asyncore.dispatcher.__init__(self)
        print("Launching Realsense Camera Server")
        try:
            self.pipeline = openPipeline()
        except:
            print("Unexpected error: ", sys.exc_info()[1])
            sys.exit(1)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        print('sending acknowledgement to', address)
        
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.w, self.h = depth_profile.width(), depth_profile.height()
        '''
        depth_intrinsics = depth_profile.get_intrinsics()  
        self.w, self.h = depth_intrinsics.width, depth_intrinsics.height
        '''
        # Processing blocks: reduce the resolution of the depth image using post processing
        self.pc = rs.pointcloud()
        
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2)
        '''
         decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)   
        # state.decimate==1 when initializing, and server initializes once at worktime (I hope so)
        '''
        self.colorizer = rs.colorizer()

        self.out = np.empty((self.h, self.w, 3), dtype=np.uint8)

        self.frame_data = ''
        self.connect((address[0], 1024))
        self.packet_id = 0 

    def handle_connect(self):
        print("connection received")

    def writable(self):
        return True

    def update_frame(self):
        depth_frame, timestamp = getDepthAndTimestamp(self.pipeline, self.decimate_filter)
        '''
        depth_image, color_image, timestamp, self.w, self.h = getDepthAndTimestamp(self.pipeline, self.decimate_filter, self.w, self.h) 
        if (depth_image is not None) and (color_image is not None): 
        '''
        if depth_frame is not None:
                self.color_source = np.asanyarray(
                    self.colorizer.colorize(depth_frame).get_data()
                )  # depth_colormap

                # convert the depth image to a string for broadcast
                data = pickle.dumps(np.asanyarray(depth_frame.get_data()))
                # capture the lenght of the data portion of the message	
                length = struct.pack('<I', len(data))
                # include the current timestamp for the frame
                ts = struct.pack('<d', timestamp)
                # for the message for transmission
                self.frame_data = b''.join([length, ts, data])

    def handle_write(self):
	    # first time the handle_write is called
        if not hasattr(self, 'frame_data'):
            self.update_frame()
	    # the frame has been sent in it entirety so get the latest frame
        if len(self.frame_data) == 0:
            self.update_frame()
        else:
	        # send the remainder of the frame_data until there is no data remaining for transmition
            remaining_size = self.send(self.frame_data)
            self.frame_data = self.frame_data[remaining_size:]
	
    def handle_close(self):
        self.pipeline.stop()
        self.close()
            

class MulticastServer(asyncore.dispatcher):
    def __init__(self, host = mc_ip_address, port=1024):
        asyncore.dispatcher.__init__(self)
        server_address = ('', port)
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bind(server_address) 	

    def handle_read(self):
        data, addr = self.socket.recvfrom(42)
        print('Received Multicast message %s bytes from %s' % (data, addr))
	    # Once the server recives the multicast signal, open the frame server
        EtherSenseServer(addr)
        print(sys.stderr, data)

    def writable(self): 
        return False # don't want write notifies

    def handle_close(self):
        self.close()

    def handle_accept(self):
        channel, addr = self.accept()
        print('received %s bytes from %s' % (data, addr))


def main(argv):
    # initalise the multicast receiver 
    server = MulticastServer()
    # hand over excicution flow to asyncore
    asyncore.loop()
   
if __name__ == '__main__':
    main(sys.argv[1:])
