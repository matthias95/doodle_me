# doodle_me

Sketch-like image stylization using webassembly.

**Live Demo:** https://matthias95.github.io/doodle_me/ 

# Usage
The repo contains a `Dockerfile` with all dependencies required to compile webassembly. `./dockerenv.sh` build the docker image and executes all supplied commands within in. 
### Run demo in local webserver
`./dockerenv.sh make serve`

### Build
`./dockerenv.sh make c_functions.js`

# Samples
<p float="left">
  <img src="img/20210522_110456.jpg" width="49%">
  <img src="img/2021_05_22_09_06_40.png" width="49%">
</p>

<p float="left">
  <img src="img/20210522_110529.jpg" width="49%">
  <img src="img/2021_05_22_09_18_24.png" width="49%">
</p>



