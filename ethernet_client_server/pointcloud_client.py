'''
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
'''

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs2
import sys, getopt
import asyncore
import pickle
import socket
import struct



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
        asyncore.loop()

        # Look for responses from all recipients
        
    except socket.timeout:
        print("timed out, no more responses")
    finally:
        print(sys.stderr, "closing socket")
        sock.close()


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = "RealSense L515"  
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

    
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


#UDP client for each camera server 
class ImageClient(asyncore.dispatcher):
    def __init__(self, server, source):   
        asyncore.dispatcher.__init__(self, server)
        self.address = server.getsockname()[0]
        self.port = source[1]
        self.buffer = bytearray()

        cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(state.WIN_NAME, self.mouse_cb)

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
        if not state.paused:
            received_data = pickle.loads(self.buffer)
            depth_image = received_data['depth_image']
            intrinsics = {
                'w':      received_data['width'],
                'h':      received_data['height'],
                'fx':     received_data['fx'],
                'fy':     received_data['fy'],
                'ppx':    received_data['ppx'],
                'ppy':    received_data['ppy'],
                'coeffs': received_data['coeffs'],
                'model':  received_data['model']
            }
            # big_depth_frame = cv2.resize(depth_frame, (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST) 

            self.out = np.empty((intrinsics['h'], intrinsics['w'], 3), dtype=np.uint8)

            self.depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            decimate = rs2.decimation_filter()
            decimate.set_option(rs2.option.filter_magnitude, 2 ** state.decimate)
            '''
            pc = rs2.pointcloud()
            self.points = pc.calculate(self.depth_frame)
            pc.map_to(self.depth_frame)  
            '''
            self.points = self.create_point_cloud(depth_image, intrinsics)
            self.render(intrinsics)

            self.buffer = bytearray()
            self.frame_id += 1

    def create_point_cloud(self, depth_image, intrinsics):
        h, w = intrinsics['h'], intrinsics['w']
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        ppx = intrinsics['ppx']
        ppy = intrinsics['ppy']

        points = []

        # walk through each pixel of depth image
        for h_pixel in range(h):
            for w_pixel in range(w):
                z = depth_image[h_pixel, w_pixel] 
                x = (w_pixel - ppx) * z / fx
                y = (h_pixel - ppy) * z / fy
                
                points.append([x, y, z])

        return np.array(points)

    
    def readable(self):
        return True


    # Viewer

    def mouse_cb(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:
            h, w = self.out.shape[:2]
            dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

            if state.mouse_btns[0]:
                state.yaw += float(dx) / w * 2
                state.pitch -= float(dy) / h * 2

            elif state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                state.translation -= np.dot(state.rotation, dp)

            elif state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                state.translation[2] += dz
                state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            state.translation[2] += dz
            state.distance -= dz

        state.prev_mouse = (x, y)


    def project(self, v):
        """project 3d vector array to 2d"""
        h, w = self.out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * (w * view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj


    def view(self, v):
        """apply view transformation on vector array"""
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


    def line3d(self, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = self.project(pt1.reshape(-1, 3))[0]
        p1 = self.project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, self.out.shape[1], self.out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(self.out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(self, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            self.line3d(
                self.view(pos + np.dot((x, 0, -s2), rotation)),
                self.view(pos + np.dot((x, 0, s2), rotation)), 
                color)
        for i in range(0, n+1):
            z = -s2 + i*s
            self.line3d(
                self.view(pos + np.dot((-s2, 0, z), rotation)),
                self.view(pos + np.dot((s2, 0, z), rotation)), 
                color)


    def axes(self, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(
            pos, 
            pos + np.dot((0, 0, size), rotation), 
            (0xff, 0, 0), 
            thickness)
        self.line3d(
            pos, 
            pos + np.dot((0, size, 0), rotation), 
            (0, 0xff, 0), 
            thickness)
        self.line3d(
            pos, 
            pos + np.dot((size, 0, 0), rotation), 
            (0, 0, 0xff), 
            thickness)


    def frustum(self, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view([0, 0, 0])
        w, h = intrinsics['w'], intrinsics['h'] 

        for d in range(1, 6, 2):
            def deproject_pixel_to_point(intrin, pixel, depth):
                x = (pixel[0] - intrin['ppx']) / intrin['fx']
                y = (pixel[1] - intrin['ppy']) / intrin['fy']
                if intrinsics['model'] == 'RS2_DISTORTION_INVERSE_BROWN_CONRADY':
                    r2 = x*x + y*y
                    f = 1 + intrin['coeffs'][0]*r2 + intrin['coeffs'][1]*r2*r2 + intrin['coeffs'][4]*r2*r2*r2
                    ux = x*f + 2*intrin['coeffs'][2]*x*y + intrin['coeffs'][3]*(r2 + 2*x*x)
                    uy = y*f + 2*intrin['coeffs'][3]*x*y + intrin['coeffs'][2]*(r2 + 2*y*y)
                    x = ux
                    y = uy
                return [depth * x, depth * y, depth]

            def get_point(x, y):
                p = deproject_pixel_to_point(intrinsics, [x, y], d)  # get inside to replace with custom function
                self.line3d(self.out, orig, self.view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            self.line3d(self.view(top_left), self.view(top_right), color)
            self.line3d(self.view(top_right), self.view(bottom_right), color)
            self.line3d(self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(self.view(bottom_left), self.view(top_left), color)


    def pointcloud(self, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(v[s])
        else:
            proj = self.project(self.view(verts))

        if state.scale:
            proj *= 0.5**state.decimate

        h, w = self.out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        self.out[i[m], j[m]] = color[u[m], v[m]]
    
    
    
    def render(self, intrinsics):   # Render
        now = time.time()

        w, h = intrinsics['w'], intrinsics['h']

        self.out.fill(0)

        self.grid((0, 0.5, 1), size=1, n=10)
        self.frustum(intrinsics)
        self.axes(self.view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        verts = np.asanyarray(
            self.points#.get_vertices()
        ).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(
            self.points#.get_texture_coordinates()
        ).view(np.float32).reshape(-1, 2)  # uv 

        
        if not state.scale or self.out.shape[:2] == (h, w):
            self.pointcloud(verts, texcoords, self.depth_colormap)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            self.pointcloud(tmp, verts, texcoords, self.depth_colormap)
            tmp = cv2.resize(
                tmp, self.out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(self.out, tmp > 0, tmp)

        if any(state.mouse_btns):
            self.axes(self.view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now

        cv2.setWindowTitle(
            state.WIN_NAME, 
            "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else "")
        )

        cv2.imshow(state.WIN_NAME, self.out)
        key = cv2.waitKey(1)

        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("d"):
            state.decimate = (state.decimate + 1) % 3
            self.decimate.set_option(rs2.option.filter_magnitude, 2 ** state.decimate)

        if key == ord("z"):
            state.scale ^= True

        if key == ord("c"):
            state.color ^= True

        if key == ord("s"):
            cv2.imwrite('./out.png', self.out)

        if key == ord("e"):
            self.points.export_to_ply('./out.ply', self.depth_frame)

        # if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        #     self.



state = AppState()

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))
mc_ip_address = "172.20.10.10"
local_ip_address = "172.20.10.2"
port = 1024
chunk_size = 4096
w, h = 640, 480


def main(argv):
    multi_cast_message(mc_ip_address, port, "EtherSensePing")
    
if __name__ == '__main__':
    main(sys.argv[1:])
                        