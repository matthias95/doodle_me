from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl
import os 

os.chdir('./html')

SimpleHTTPRequestHandler.extensions_map['.wasm'] = 'application/wasm'
httpd = HTTPServer(('0.0.0.0', 8000), SimpleHTTPRequestHandler)

httpd.socket = ssl.wrap_socket (httpd.socket, 
        keyfile='../sslcert/server.key', 
        certfile='../sslcert/server.crt', server_side=True)
print('Lauch server at: https://127.0.0.1:8000/')
httpd.serve_forever()