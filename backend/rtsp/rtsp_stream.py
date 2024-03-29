import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)


class MyFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, stream_mp4_path=None, **properties):
        super(MyFactory, self).__init__(**properties)
        self.mp4_file_path = stream_mp4_path
        self.launch_string = f'( filesrc location="{self.mp4_file_path}" ! qtdemux ! h264parse ! rtph264pay name=pay0 pt=96 )'

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)


class GstServer:
    def __init__(
        self,
        ip="127.0.0.1",
        port="8554",
        stream_url="/stream",
        stream_mp4_path=None,
    ):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_address(ip)
        self.server.set_service(port)
        self.factory = MyFactory(stream_mp4_path)
        self.factory.set_shared(True)
        self.mount_points = self.server.get_mount_points()
        self.mount_points.add_factory(stream_url, self.factory)

        self.server.attach(None)
        print(
            f"RTSP server started at rtsp://10.28.224.201:{port}{stream_url}"
        )


if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 30437
    stream_url = "/stream/cctv"
    stream_mp4_path = "Abnormal_2_DYA_valid_2hour.mp4"
    s = GstServer(
        ip=ip,
        port=str(port),
        stream_url=stream_url,
        stream_mp4_path=stream_mp4_path,
    )

    loop = GLib.MainLoop()
    loop.run()
