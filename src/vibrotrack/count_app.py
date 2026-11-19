"""Loopback-only single-image bench UI; images stay in memory."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import cv2
from PIL import Image

from . import count

MAX_UPLOAD = 64 * 1024 * 1024
CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
       "img-src data: blob:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")


def make_server(*, cfg=None, mask_path=None):
    page = count.viewer_page(None).encode('utf-8')
    worker = Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(30)
            super().setup()

        def log_message(self, format, *args):
            pass

        def reply(self, status, body, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', CSP)
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, data):
            self.reply(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                       'application/json; charset=utf-8')

        def allowed(self, post=False):
            host = f'127.0.0.1:{self.server.server_port}'
            return (self.headers.get('Host') == host and
                    (not post or self.headers.get('Origin') == 'http://' + host))

        def do_GET(self):
            if not self.allowed():
                self.send_json(403, {'error': '仅支持通过本机地址访问'})
            elif self.path == '/':
                self.reply(200, page, 'text/html; charset=utf-8')
            else:
                self.send_json(404, {'error': '页面不存在'})

        def do_POST(self):
            if not self.allowed(post=True):
                self.send_json(403, {'error': '仅允许本机页面发起检测'})
                return
            url = urlsplit(self.path)
            if url.path != '/detect':
                self.send_json(404, {'error': '接口不存在'})
                return
            query = parse_qs(url.query)
            mode = query.get('mode', [''])[0]
            if mode not in count.MODES:
                self.send_json(400, {'error': '请先选择粒径模式'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_UPLOAD or self.headers.get_content_type() != 'application/octet-stream':
                self.send_json(400, {'error': '请选择一张不超过 64 MiB 的图片'})
                return
            name = query.get('name', ['image'])[0].replace('\\', '/').rsplit('/', 1)[-1][:240]
            if not worker.acquire(blocking=False):
                self.send_json(409, {'error': '正在检测另一张图片，请稍后再试'})
                return
            try:
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError('图片读取不完整')
                det, image, meta = count.count_bytes(raw, name, mode, cfg, mask_path)
                data = count.viewer_data(image, det, meta)
            except (cv2.error, Image.DecompressionBombError):
                self.send_json(422, {'error': '无法处理这张图片，请确认格式和尺寸'})
                return
            except (ValueError, OSError) as exc:
                self.send_json(422, {'error': str(exc) if not isinstance(exc, Image.UnidentifiedImageError)
                                     else '无法识别图片，请使用 TIFF、PNG 或 JPEG'})
                return
            finally:
                worker.release()
            self.send_json(200, data)

    return ThreadingHTTPServer(('127.0.0.1', 0), Handler)


def run(*, cfg=None, mask_path=None, open_browser=True):
    with make_server(cfg=cfg, mask_path=mask_path) as server:
        url = f'http://127.0.0.1:{server.server_port}/'
        print(f'颗粒速查：{url}（Ctrl+C 退出）', flush=True)
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0
