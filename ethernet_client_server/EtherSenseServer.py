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
    # take ownership of the frame for further processing
    frames.keep()
    depth = frames.get_depth_frame()

    if depth:
        depth_proc = depth_filter.process(depth)
        # take ownership of the frame for further processing
        depth_proc.keep()
        # represent the frame as a numpy array
        depthData = depth_proc.as_frame().get_data()        
        depthMat = np.asanyarray(depthData)

        ts = frames.get_timestamp()

        return depthMat, ts
    
    else:
        return None, None


def openPipeline():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    pipeline_profile = pipeline.start(cfg)
    sensor = pipeline_profile.get_device().first_depth_sensor()
    print("Depth sensor available:", sensor)
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
        print('Opened socket. Sending acknowledgement to', address)
        
	    # reduce the resolution of the depth image using post processing
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2)
        self.frame_data_depth = ''
        self.connect((address[0], 1024))
        self.packet_id = 0        

    def handle_connect(self):
        print("Connection received")

    def writable(self):
        return True

    def update_frame(self):
        depth, timestamp = getDepthAndTimestamp(self.pipeline, self.decimate_filter)
        if depth is not None:
            # convert the depth image to a string for broadcast
                depth_data = pickle.dumps(depth)
            # capture the lenght of the data portion of the message	
                depth_length = struct.pack('<I', len(depth_data))
            # include the current timestamp for the frame
                ts = struct.pack('<d', timestamp)
            # for the message for transmission
                self.frame_data_depth = b''.join([depth_length, ts, depth_data])

    def handle_write(self):
	    # first time the handle_write is called
        if not hasattr(self, 'frame_data_depth'):
            self.update_frame()
	    # the frame has been sent in it entirety so get the latest frame
        if not len(self.frame_data_depth):
            self.update_frame()
        else:
	        # send the remainder of the frame_data until there is no data remaining for transmition
            remaining_size_d = self.send(self.frame_data_depth)
            self.frame_data_depth = self.frame_data_depth[remaining_size_d:]
	
    def handle_close(self):
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
        print('Received %s bytes from %s' % (data, addr))


def main(argv):
    # initalise the multicast receiver 
    server = MulticastServer()
    # hand over excicution flow to asyncore
    asyncore.loop()
   
if __name__ == '__main__':
    main(sys.argv[1:])
