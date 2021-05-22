SHELL=/bin/bash

docker_image: Dockerfile
	docker build -t doodle_me .
	docker image prune -f
	touch docker_image

default: c_functions.js

c_functions.exe: c_functions.cpp
	g++ $< -std=c++17  -fopenmp -o $@

c_functions.so: c_functions.cpp
	g++ $< -std=c++17 -Ofast -fopenmp -fPIC -shared -o $@

c_functions.js: c_functions.cpp
	em++ $<  -O3 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1  -o $@  -s "EXPORTED_FUNCTIONS=['_malloc', '_free', '_rasterize_img_tiled_uint8', '_rgb2gray']" -std=c++17  --llvm-lto 3
	mv -f ./c_functions.js ./html
	mv -f ./c_functions.wasm ./html

run: c_functions.exe c_functions.so 
	./c_functions.exe

sslcert:
	mkdir -p sslcert
	openssl req -newkey rsa:4096 -x509 -sha256 -days 3650 -nodes -out sslcert/server.crt -keyout sslcert/server.key -batch
	
serve:  sslcert c_functions.js
	python3 index.py

publish:
	cp -r html doc

clean:
	rm -f c_functions.exe
	rm -f c_functions.so
	rm -f c_functions.dll 
	rm -f c_functions.wasm 
	rm -f c_functions.js
	rm -f c_functions.s
	rm -f html/c_functions.js
	rm -f html/c_functions.wasm 
	rm -rf venv
	rm -f docker_image

