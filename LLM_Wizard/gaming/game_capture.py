import os
import signal
import time
import uuid
from typing import List, Optional, Deque

import dbus
import numpy as np
from PIL import Image
from dbus.mainloop.glib import DBusGMainLoop

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from collections import deque
from queue import Empty
import multiprocessing as mp

# Initialize GStreamer and DBus loop
Gst.init(None)
DBusGMainLoop(set_as_default=True)

class ScreenCastCapture:
    def __init__(self, source_type: int = 1, cursor_mode: int = 2):
        """
        source_type: 1 (monitor/full-screen), 2 (window)
        cursor_mode: 1 (hidden), 2 (embedded), 4 (metadata)
        """
        self.bus = dbus.SessionBus()
        self.portal = self.bus.get_object('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop')
        self.portal_interface = dbus.Interface(self.portal, 'org.freedesktop.portal.ScreenCast')
        self.request_interface = 'org.freedesktop.portal.Request'
        self.session = None
        self.pipeline = None
        self.node_id = None
        self.source_type = source_type
        self.cursor_mode = cursor_mode
        self.loop = GLib.MainLoop()
        self.response_handlers = {}
        self.request_path_map = {} # To map request paths back to tokens
        signal.signal(signal.SIGINT, self.stop_capture)

    def _generate_token(self) -> str:
        return str(uuid.uuid4()).replace('-', '_')

    def _request_path(self, token: str) -> str:
        unique_name = self.bus.get_unique_name()[1:].replace('.', '_')
        path = f'/org/freedesktop/portal/desktop/request/{unique_name}/{token}'
        self.request_path_map[path] = token # Map path to token
        return path

    def _add_response_handler(self, token: str, callback):
        path = self._request_path(token)
        self.response_handlers[token] = callback
        self.bus.add_signal_receiver(
            self._handle_response,
            signal_name='Response',
            dbus_interface=self.request_interface,
            path=path,
            path_keyword='path' # Ask dbus-python to pass the object path
        )

    def _handle_response(self, response_code: int, results: dict, path=None):
        if path and path in self.request_path_map:
            token = self.request_path_map[path]
            if token in self.response_handlers:
                callback = self.response_handlers[token]
                if response_code != 0:
                    print(f"Portal request failed for token {token}: code {response_code}, results {results}")
                    self.loop.quit()
                    return

                callback(results)
                # Clean up
                del self.response_handlers[token]
                del self.request_path_map[path]
        else:
            print(f"Warning: Received a response for an unknown request path: {path}")


    def create_session(self, callback):
        token = self._generate_token()
        self._add_response_handler(token, lambda res: callback(res['session_handle']))
        options = {'handle_token': token, 'session_handle_token': self._generate_token()}
        self.portal_interface.CreateSession(options)

    def select_sources(self, session_handle: str, callback):
        token = self._generate_token()
        # The response for SelectSources has no results, so we just need to trigger the next step.
        self._add_response_handler(token, lambda _: callback())
        options = {
            'handle_token': token,
            'types': dbus.UInt32(self.source_type),
            'cursor_mode': dbus.UInt32(self.cursor_mode),
            'multiple': False,
        }
        self.portal_interface.SelectSources(session_handle, options)

    def start_session(self, session_handle: str, callback):
        token = self._generate_token()
        self._add_response_handler(token, lambda res: callback(res['streams']))
        options = {'handle_token': token}
        self.portal_interface.Start(session_handle, '', options)

    def setup_pipeline(self, node_id: int, fd: int):
        print(f"Setting up pipeline with PipeWire node ID: {node_id} and FD: {fd}")
        pipeline_str = (
            f'pipewiresrc fd={fd} path={node_id} do-timestamp=true ! '
            'videoconvert ! video/x-raw,format=RGB ! '
            'appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline state set to PLAYING.")
        # A small delay to allow the pipeline to start producing samples
        GLib.timeout_add(500, self.loop.quit) # Quit the loop to start capturing

    def capture_frame(self) -> Optional[np.ndarray]:
        if not self.pipeline or self.pipeline.get_state(0)[1] != Gst.State.PLAYING:
            print("Pipeline not ready or not playing.")
            return None
        sink = self.pipeline.get_by_name('sink')
        sample = sink.emit('pull-sample')
        if not sample:
            print("Failed to pull sample from appsink.")
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        height, width = caps.get_value('height'), caps.get_value('width')
        data = buf.extract_dup(0, buf.get_size())
        return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

    def start_capture(self):
        def on_create(session_handle):
            print(f"Session created: {session_handle}")
            self.session = session_handle
            # Pass a lambda that calls start_session as the callback
            self.select_sources(session_handle, lambda: self.start_session(session_handle, on_start))

        def on_start(streams):
            print(f"Session started, streams: {streams}")
            if not streams:
                print("No streams available.")
                self.loop.quit()
                return
            stream_props = streams[0] # This is a dbus.Struct
            self.node_id = stream_props[0]
            
            # The key change is here!
            # The file descriptor is passed in the 'options' dictionary of the method call.
            # We need to tell dbus-python to expect it.
            fd_obj = self.portal_interface.OpenPipeWireRemote(self.session, {}, get_handles=True)
            
            # The returned object is now a dbus.UnixFd
            # We extract the integer file descriptor from it.
            fd = fd_obj.take()
            
            self.setup_pipeline(self.node_id, fd)

        self.create_session(on_create)
        print("Starting GLib MainLoop for setup. A portal dialog should appear.")
        print("Please approve the screen sharing request.")
        self.loop.run()

    def stop_capture(self, *args):
        print("Stopping capture...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.session:
            try:
                session_iface = dbus.Interface(self.bus.get_object('org.freedesktop.portal.Desktop', self.session), 'org.freedesktop.portal.Session')
                session_iface.Close()
            except dbus.exceptions.DBusException as e:
                print(f"Could not close session (it may already be closed): {e}")
        if self.loop.is_running():
            self.loop.quit()

def capture_images(images_dequeue: Deque, interval_sec: float = 0.1, source_type: int = 1):
    capturer = ScreenCastCapture(source_type=source_type)
    try:
        # This part now blocks until the user approves the portal and the pipeline is ready
        capturer.start_capture()
        while True:
            frame = capturer.capture_frame()
            if frame is not None:
                print(f"  - Captured frame")
                images_dequeue.append(Image.fromarray(frame))
            else:
                print(f"  - Failed to capture frame")
            time.sleep(interval_sec)
    except Exception as e:
        print(f"An error occurred during capture: {e}")
    finally:
        capturer.stop_capture()




class GameCaptureWorker(mp.Process):
    """
    An independent worker process for capturing the screen.

    This worker is designed to run in its own process to avoid blocking the
    main application and to bypass the GIL. It communicates with the main
    process via multiprocessing queues.
    """
    def __init__(self,
                 image_data_queue: mp.Queue,
                 command_queue: mp.Queue,
                 stop_event: mp.Event,
                 source_type: int = 1,
                 interval_sec: float = 0.1,
                 target_size: Optional[tuple[int, int]] = (1280, 720)):
        super().__init__()
        self.image_data_queue = image_data_queue
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.source_type = source_type
        self.interval_sec = interval_sec
        self.target_size = target_size
        self.capturer = None

    def run(self):
        """The main loop of the worker process."""
        print(f"[CaptureWorker-{os.getpid()}] Starting...")
        # GStreamer and DBus objects must be instantiated within the new process
        self.capturer = ScreenCastCapture(source_type=self.source_type)
        
        try:
            # This blocks until the user approves the portal and the pipeline is ready
            self.capturer.start_capture()

            while not self.stop_event.is_set():
                start_time = time.perf_counter()

                self._handle_commands()

                frame = self.capturer.capture_frame()
                if frame is not None:
                    image = Image.fromarray(frame)

                    # Pre-process the image (resize) before sending
                    # This reduces data transfer size and offloads work from the main process
                    if self.target_size:
                        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

                    # Send serialized data for efficiency
                    # A dictionary of primitives is much faster to pickle than a PIL object
                    data_packet = {
                        'image_bytes': image.tobytes(),
                        'size': image.size,
                        'mode': image.mode,
                    }
                    self.image_data_queue.put(data_packet)
                
                # Maintain the desired interval
                elapsed = time.perf_counter() - start_time
                sleep_time = self.interval_sec - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"[CaptureWorker-{os.getpid()}] An error occurred: {e}")
        finally:
            print(f"[CaptureWorker-{os.getpid()}] Stopping...")
            if self.capturer:
                self.capturer.stop_capture()

    def _handle_commands(self):
        """Check for and execute commands from the main process."""
        try:
            command = self.command_queue.get_nowait()
            if command['action'] == 'set_interval':
                new_interval = command['value']
                print(f"[CaptureWorker-{os.getpid()}] Updating interval to {new_interval}s")
                self.interval_sec = new_interval
        except Empty:
            pass # No commands