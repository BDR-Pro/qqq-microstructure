# Part of qqq-microstructure.
#
# A stand-in for llama.cpp's llama-server, for tests/test_decide_local.py: it
# takes the same flags decide_local passes, reads the key from LLAMA_API_KEY,
# and speaks just enough of /health, /v1/models and /v1/systemone. STUB_*
# environment variables make it misbehave the ways the real one could.

import os, sys, json, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument('-m')
ap.add_argument('--alias')
ap.add_argument('--host')
ap.add_argument('--port', type=int)
ap.add_argument('-c')
ap.add_argument('-np')
ap.add_argument('-t')
a = ap.parse_args()
KEY = os.environ.get('LLAMA_API_KEY')
ALIAS = os.environ.get('STUB_ALIAS', a.alias)
NOAUTH = os.environ.get('STUB_NOAUTH') == '1'      # no key needed
ANYKEY = os.environ.get('STUB_ANYKEY') == '1'      # any key accepted
MODEL = os.environ.get('STUB_MODEL')               # answers under this name
P = float(os.environ.get('STUB_P', '0.6'))
if os.environ.get('STUB_CONFIG') == '1':           # as llama-server logs it
    print('using config file: /etc/llama.cpp/config.ini', flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def authed(self):
        got = self.headers.get('Authorization')
        return NOAUTH or (ANYKEY and bool(got)) or got == f'Bearer {KEY}'

    def do_GET(self):
        if self.path == '/health':
            return self.send(200, {'status': 'ok'})
        if self.path == '/v1/models':
            if not self.authed():
                return self.send(401, {'error': {'code': 401}})
            return self.send(200, {'data': [{'id': ALIAS, 'architecture': {
                'output_modalities': ['decisions']}}]})
        self.send(404, {})

    def do_POST(self):
        if not self.authed():
            return self.send(401, {'error': {'code': 401}})
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ans = {k: {'type': 'noul', 'noul': P} for k in body['questions']}
        self.send(200, {'model': MODEL or ALIAS, 'answers': ans})


ThreadingHTTPServer((a.host, a.port), H).serve_forever()
