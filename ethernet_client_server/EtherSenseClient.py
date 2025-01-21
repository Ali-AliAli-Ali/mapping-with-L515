#!/usr/bin/python
import pyrealsense2 as rs
import sys, getopt
import asyncore
import numpy as np
import pickle
import socket
import struct
import cv2


print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))
mc_ip_address = "172.20.10.10"
local_ip_address = "172.20.10.2"
port = 1024
chunk_size = 4096

def main(argv):
    multi_cast_message(mc_ip_address, port, "EtherSensePing")
        

#UDP client for each camera server 
class ImageClient(asyncore.dispatcher):
    def __init__(self, server, source):   
        asyncore.dispatcher.__init__(self, server)
        self.address = server.getsockname()[0]
        self.port = source[1]
        self.buffer = bytearray()
        self.windowName = self.port
        # open cv window which is unique to the port 
        # and named after port
        cv2.namedWindow("window of port" + str(self.windowName))
        self.remainingBytes = 0
        self.frame_id = 0
       
    def handle_read(self):
        # start reading new frame
        if self.remainingBytes == 0:  
            # get the expected frame size
            self.frame_length = struct.unpack('<I', self.recv(4))[0]
            # get the timestamp of the current frame
            self.timestamp = struct.unpack('<d', self.recv(8))
            self.remainingBytes = self.frame_length
        
        # request the frame data until the frame is completely in buffer
        data = self.recv(self.remainingBytes)
        self.buffer += data
        self.remainingBytes -= len(data)
        # once the frame is fully recived, process/display it
        if len(self.buffer) == self.frame_length:
            self.handle_frame()

    def handle_frame(self):
        # convert the frame from string to numerical data
        # depth_image = pickle.loads(self.buffer)
        received_data = pickle.loads(self.buffer)    
        depth_image = received_data['depth_image']    

        big_depth_image = cv2.resize(depth_image, (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST) 

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(big_depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.putText(depth_colormap, str(self.timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("window of port" + str(self.windowName), depth_colormap)
        cv2.waitKey(1)
        self.buffer = bytearray()
        self.frame_id += 1

    def readable(self):
        return True

    
class EtherSenseClient(asyncore.dispatcher):
    def __init__(self):
        asyncore.dispatcher.__init__(self)
        self.server_address = ('', 1024)
        # create a socket for TCP connection between the client and server
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)
        
        self.bind(self.server_address) 	
        self.listen(10)

    def writable(self): 
        return False # don't want write notifies

    def readable(self):
        return True
        
    def handle_connect(self):
        print("connection received")

    def handle_accept(self):
        pair = self.accept()
        # print(self.recv(10))
        if pair is not None:
            sock, addr = pair
            print("Incoming connection from %s" % repr(addr))
            # when a connection is attempted, delegate image receival to the ImageClient 
            handler = ImageClient(sock, addr)

def multi_cast_message(ip_address, port, message):
    # send the multicast message
    multicast_group = (ip_address, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    connections = {}
    try:
        # Send data to the multicast group
        print("sending '%s'" % message + str(multicast_group))
        sent = sock.sendto(message.encode(), multicast_group)
   
        # defer waiting for a response using Asyncore
        client = EtherSenseClient()
        # start channel services: asyncore.dispather (EtherSenseClient) as well.
        # 1. get to readable, writable methods
        # 2. get to handle_accept
        asyncore.loop()

        # Look for responses from all recipients
        
    except socket.timeout:
        print("timed out, no more responses")
    finally:
        print(sys.stderr, "closing socket")
        sock.close()

if __name__ == '__main__':
    main(sys.argv[1:])