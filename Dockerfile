FROM ubuntu

RUN apt update &&\
    apt install -qq -y python3 &&\
    apt install -qq -y cmake &&\
    apt install -qq -y git &&\
    apt install -qq -y nano &&\
    apt install -qq -y openssl &&\
    apt install -qq -y g++ &&\
    rm -rf /var/lib/apt/lists/* &&\
    git clone https://github.com/emscripten-core/emsdk.git &&\
    cd emsdk &&\
    git pull &&\
    ./emsdk install latest &&\
    ./emsdk activate latest &&\
    chmod +x /emsdk/emsdk_env.sh

RUN echo "source /emsdk/emsdk_env.sh" >> /etc/bash.bashrc

RUN echo "#!/bin/bash" >> run.sh &&\ 
    echo 'source /emsdk/emsdk_env.sh &&  "$@"' >> run.sh &&\
    chmod +x run.sh

WORKDIR /app
ENTRYPOINT ["/run.sh"]