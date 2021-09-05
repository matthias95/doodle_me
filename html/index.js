function rasterize_img(image_data_source, image_data_destination) {
    const bufferPointerIn = Module._malloc(image_data_source.width * image_data_source.height * 4);
    const bufferPointerOut = Module._malloc(image_data_destination.width * image_data_destination.height * 4);
    const bufferIn = new Uint8Array(Module.HEAPU8.buffer, bufferPointerIn, image_data_source.width * image_data_source.height * 4);    
    bufferIn.set(image_data_source.data);
    
    let max_val = Number($('#max_val_slider').val())/100.0;
    let marker_size = (Number($('#marker_size_slider').val())/100.0)*20 * (getSelectedSize() / (256*3));
    
    wasmModule._rasterize_img_tiled_uint8(bufferPointerIn, image_data_source.height, image_data_source.width, bufferPointerOut, image_data_destination.height, image_data_destination.width, max_val, marker_size, 80, 40);
    
    const bufferOut = new Uint8Array(Module.HEAPU8.buffer, bufferPointerOut, image_data_destination.width * image_data_destination.height * 4);
    image_data_destination.data.set(bufferOut);
    Module._free(bufferPointerIn);
    Module._free(bufferPointerOut);
}

function renderSource(source, destination) {
    const context_source = source.getContext('2d');
    const context_destination = destination.getContext('2d');

    const image_data_source = context_source.getImageData(0, 0, source.width, source.height);
    const image_data_destination = context_destination.getImageData(0, 0, destination.width, destination.height);

    console.time('rasterize_img')
    rasterize_img(image_data_source, image_data_destination);
    console.timeEnd('rasterize_img')

    context_destination.putImageData(image_data_destination, 0, 0);
}


wasmModule = Module;
var imageCapture;
const canvas_input = document.createElement('canvas');
const loading_modal = $("#loading");

var camera_id = 0;

function getCameraStream(switch_camera=false) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    var cfg_provider = new Promise(function(resolve, reject){
      return resolve({video: true});
    });

  } else {
    var cfg_provider = navigator.mediaDevices.enumerateDevices().then(el => el.filter(dev => dev.kind == "videoinput")).then(
      cams =>{
        camera_id = camera_id % cams.length;
        if(cams.length > 1) {
          if(switch_camera){
            camera_id = (camera_id + 1) % cams.length;
          }
          var media_cfg = {video: {'deviceId':cams[camera_id].deviceId}};
        } else {
          var media_cfg = {video: true};
        }
        return media_cfg;
      }
    );
  }
  

  cfg_provider.then(cfg => {
    navigator.mediaDevices.getUserMedia(cfg)
  .then(mediaStream => {
    video = document.querySelector('video');
    video.srcObject = mediaStream;
    
    if(typeof ImageCapture != "undefined"){
      let video_tracks = mediaStream.getVideoTracks();
      const track = video_tracks[0];
      imageCapture = new ImageCapture(track);
    } else {
      imageCapture = {};
      imageCapture.takePhoto = function() {
        return new Promise(function(resolve, reject){
          return resolve(video);
        });
      };
    }
    
  })
  .catch(
      error => {
        showFileInput();
        console.log(error);
        if(error.name == "NotAllowedError") {
          alert('Webcam access is required!');
        }else if(error.name == "NotFoundError") {
          alert('No webcam found!');
        } else {
          camera_id++;
        }
      }
    );
  }
  );
}

function hideLoadingModal(_){
  loading_modal.modal('hide');
}

function showLoadingModal(){
  $('#loading').find('#staticBackdropLabel').text('Loading (can take up to 60 sconds)');
  loading_modal.modal('show');
}

function hideModal(){
  loading_modal.modal('hide');
}
function showDownloadModal(){
  $('#loading').find('#staticBackdropLabel').text('Preparing Download');
  loading_modal.modal('show');
}

function getSelectedSize() {
  switch($('input[name="select_size"]:checked').val()){
    case 'S':
      var width = 256*2;
      break;
    case 'M':
      var width = 256*3;
      break;
    case 'L':
      var width = 256*4;
      break;
    case 'XL':
      var width = 256*5;
      break
  }
  return width;
}

function processImage(imageBitmap) {
  var width = getSelectedSize();
  
  var height = imageBitmap.height * (width / imageBitmap.width);

  canvas_input.height = height;
  canvas_input.width = width;

  var f = 3;
  canvas_output = document.createElement('canvas')
  canvas_output.height = height*f;
  canvas_output.width = width*f;

  const context = canvas_input.getContext('2d');
  context.drawImage(imageBitmap, 0, 0, canvas_input.width, canvas_input.height);

  renderSource(canvas_input, canvas_output);

  canvas_output = $(canvas_output).addClass("card-img-top");

  let img_card = $(`
    <div class="row"> 
      <div class="col-md-12"> 
        <div class="card">
          <div class="card-body">
            <a class="btn btn-success btn-download">Download</a>
          </div>
        </div>
      </div>
    </div>
  `);
  
  img_card.find(".card").prepend(canvas_output);
  let button = img_card.find(".btn-download")[0];

  button.addEventListener('click', function (e) {
    showDownloadModal();
    canvas_output[0].toBlob(
      blob => {
        const hyperlink = document.createElement('a');
        hyperlink.download = new Date().toISOString().slice(0,19).replace(/-|:|T/g,"_") + '.png';
        hyperlink.href = URL.createObjectURL(blob);
        hyperlink.click();
        URL.revokeObjectURL(hyperlink.href);
        hideModal();
      },
      'image/png',
      0.8,
    );
  });
  

  $("#image_container").append(img_card); 

  loading_modal.on("shown.bs.modal", hideLoadingModal);
  loading_modal.on("hidden.bs.modal", _ => loading_modal.unbind("shown.bs.modal", hideLoadingModal));
  
  hideLoadingModal();
}


function onTakePhotoButtonClick() {
    const cfg = {imageHeight:1080,imageWidth:1920 };

  showLoadingModal();
  imageCapture.takePhoto()
  .then(blob => createImageBitmap(blob))
  .then(imageBitmap => {
    processImage(imageBitmap);
  })
  .catch(error => console.log(error)); 
}


function showFileInput(){
  $("#file_input").show();
  $("#camera_input").hide();
  $("#file_input_nav").addClass("active");
  $("#camera_input_nav").removeClass("active");
}

function showCameraInput(){
  getCameraStream();
  $("#file_input").hide();
  $("#camera_input").show();
  $("#file_input_nav").removeClass("active");
  $("#camera_input_nav").addClass("active");
}

function handleImage(e){
    showLoadingModal();
    var reader = new FileReader();
    reader.onload = function(event){
        var img = new Image();
        img.onload = function(){
          processImage(img);
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]);     
}



function intercept_paste(e) {
  if (e.clipboardData && e.clipboardData.items) {
    var items = e.clipboardData.items;
    var found_image = false;
    for (var i = 0; i < items.length; i++) {
      if (items[i].type.includes("image")) {
        var URLObj = window.URL || window.webkitURL;
        var source = URLObj.createObjectURL(items[i].getAsFile());
        
        showLoadingModal();
        var img = new Image();
        img.onload = function () {
          processImage(img);
        };
        img.src = source;

        found_image = true;
      }
    }
    if(found_image){
      e.preventDefault();
    }
  }
};
document.addEventListener('paste', intercept_paste, false);

var imageLoader = document.getElementById('imageLoader');
imageLoader.addEventListener('change', handleImage, false);
showFileInput();


