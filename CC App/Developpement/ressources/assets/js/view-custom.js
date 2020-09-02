// Options for the observer (which mutations to observe)
const config = { attributes: true, childList: true, subtree: true };
var socket = null;
var imgList = Array();
const error_alert = function (errorTitle, errorMessage) {
  return `<div class="error-container">
            <div class="alert alert-danger" role="alert"  data-toggle="collapse" data-target="#collapseMessage" aria-expanded="false" aria-controls="collapseMessage">
            ${errorTitle}
            </div>
            <div class="card card-body" id ="collapseMessage">
                <span class="fa fa-times-circle text-danger"></span> ${errorMessage}  
            </div>
            </div>
        
        `;
};

const clearCanvasShapes = (context, fillImg) => {
  context.clearRect(0, 0, fillImg.width, fillImg.height);
  context.drawImage(fillImg, 0, 0, fillImg.width, fillImg.height);
};
// Callback function to execute when mutations are observed
const callback = function (mutationsList, observer) {
  var  tang=null,
  b=null,
  canvas=null;
  var point1 = null,
      point2 = null;
  for (let mutation of mutationsList) {
    if (mutation.type === "childList" || mutation.type === "subtree") {
      // Toggle Loading spinner
    
      for (let addNode of mutation.addedNodes) {
        $("#upload-loading").remove();
        if ($(addNode).find('h5,img').length){
          $("#drop-div").removeClass("d-none");
        }
        
          // Process images button Capture
        var processbtn = $(addNode).find("#process-imgs-button");
        if (processbtn.length) {
          
          // if (addNode.id === "process-imgs-button") {
          console.log("Button targeted");

          processbtn.click(function (e) {
            if ($("#usage-switch").hasClass("toggled-on")) {
              // console.log("Clicked");
              $('#output-image-process').focus();
              send_to_server();
            }
          });
        }
          // Process video button Capture
        var processVideobtn = $(addNode).find("#process-video-button");
        if (processVideobtn.length) {
          
          // if (addNode.id === "process-imgs-button") {
          console.log("Video Button targeted");

          processVideobtn.click(function (e) {
            if($('#process-video-output-flow').length)
              $("#process-video-output-flow").show();
            if ($("#usage-switch").hasClass("toggled-on")) {
              
              send_video_to_server();
            }
          });
        }

        var drawVLineCanvasBtn = $(addNode).find("#line-canvas-button");
        if (drawVLineCanvasBtn.length) {
          // if (addNode.id === "line-canvas-button") {
          console.log("captured");
          // Draw current scene on the canvas from the frame
          canvas = document.querySelector("#scene-canvas");
          // var canvas = document.createElement("canvas");
          var ctx = canvas.getContext("2d");
          var currentImg = document.querySelector("#process-video-output-flow");
          // Define Click handler
          $(drawVLineCanvasBtn).click(function (e) {
            $("#edit-canvas-area").removeClass("d-none");

            canvas.height = currentImg.height;
            canvas.width = currentImg.width;
            ctx.drawImage(currentImg,0,0,currentImg.width,currentImg.height);

           
            ctx.fillStyle = "#e31414";
            ctx.strokeStyle = "#e31414";
            ctx.lineWidth = 4;
            // Add canvas drawing handlers
            canvas.addEventListener(
              "click",
              (e) => {
                if (point1 === null) {
                  console.log('first')
                  point1 = {
                    x: e.clientX - canvas.getBoundingClientRect().x,
                    y: e.clientY - canvas.getBoundingClientRect().y,
                  };
                  return;
                } else {
                  if (point2 === null) {
                    console.log('second')
                    point2 = {
                      x: e.clientX - canvas.getBoundingClientRect().x,
                      y: e.clientY - canvas.getBoundingClientRect().y,
                    };
                    tang = (point1.y - point2.y) / (point1.x - point2.x);
                    b = point1.y - point1.x * tang;
                    let startP = {
                        x: (-1 * b) / tang,
                        y: 0,
                      },
                    endP = {
                        x: (canvas.height - 1 * b) / tang,
                        y: canvas.height,
                      };
                    ctx.beginPath();
                    ctx.moveTo(startP.x, startP.y);
                    ctx.lineTo(endP.x, endP.y);
                    ctx.stroke();
                    ctx.closePath();
                  }
                  
                }
              },
              false
            );
          });
        }
          // Get reset canvas button
        var resetCanvasBtn = $(addNode).find("#clear-canvas-button");
        if (resetCanvasBtn.length) {
            // Add click event handler
          // if (addNode.id === "clear-canvas-button") {
          $(resetCanvasBtn).click((e) => {
            point1=null;
            point2=null;
            $("#edit-canvas-area").addClass("d-none");
            // ctx.clearRect(0, 0, currentImg.width, currentImg.height);
            clearCanvasShapes(ctx, currentImg);
          });
        }
            // Get Confirm canvas draw button
        var confirmDrawBtn = $(addNode).find("#confirm-draw-btn");
        if (confirmDrawBtn.length) {
            // Add click event handler
          // if (addNode.id === "confirm-draw-btn") {
          $(confirmDrawBtn).click((e) => {
            if(point1===null || point2===null) return;
            console.log(tang,b)
            $("#edit-canvas-area").addClass("d-none");
            $("#process-video-output-flow").hide();
            $('#edit-canvas-panel').hide();
            // $('#output-video-process').append('<p id="hidden-splitLine-input" class="d-none">'+tang+'/'+b+'</p>')
            // $('#hidden-splitLine-input').get(0).innerHTML=`${tang}/${b}`;
            $.post("http://localhost:8050/scene/regions/",
              {
                tang: tang,
                b: b,
                p1: {'x': point1.x/canvas.width,'y':point1.y/canvas.height},
                p2: {'x': point2.x/canvas.width,'y':point2.y/canvas.height}
              },
              function(data, status){
                point1=null;
                point2=null;
                console.log('Received ', data, ' status', status)
            });
          });
        }
      }
    } else if (mutation.type === "attributes") {
      //console.log('The ' + mutation.attributeName + ' attribute was modified.');
    }
  }
};

const get_model_type=(model_type)=>{
  switch (model_type) {
    case "Mobile SSD":
     return "mobileSSD";
      break;
    case "YOLO":
     return "yolo";
      break;
    case "MCNN":
     return "MCNN";
      break;
    case "CSRNet":
     return "CSRNet";
      break;
    case "SANet":
     return "SANet";
      break;
    case "CCNN":
     return "CCNN";
      break;
  }
}
const instantiate_socket=(url)=>{
  socket = io.connect(url, {
    reconnection: false,
  });
  socket.on("connect", function () {
    console.log("Socket connected.");
  });
  socket.on("disconnect", function () {
    console.log("Socket disconnected.");
  });
  socket.on("server-error", function (data) {
    console.log("Server-errors occured.");
    $(document).append(
      error_alert("An error occured on the server.", data["message"])
    );
    this.disconnect();
  });
  socket.on("process-done", function (data) {
    console.log(data);
    
    // $('#output-image-process').append(x)
    errors = data["errors"];
    if (errors && errors[0]) {
      console.log("Processing frame resulted some errors.");

      $(document).append(
        error_alert("An error occured on the server.", errors[0])
      );
    }else{
      $('#server-toast .toast-body').html(`<span class="mr-3"> Process done.</span> <span class=" text-success fa fa-check-circle"></span>`);
    }
  });
  socket.on("send-image", (data) => processImageResponse(data));

    // Video process events
  socket.on("send-frame",(data)=>{
    $('#process-video-output-flow').attr('src',data['data'])
  })  
}
const send_to_server = function () {
  var url_input = $("#server-url-control input");
  if (url_input) {
    if (url_input.val()) {// If server URL is provided
      imgList = Array();
        // If socket not initialized yet, connect and create handlers.
      if (socket === null || socket.disconnected|| socket.io.uri!==url_input.val()) {
        
        instantiate_socket(url_input.val());
      }
     
      var imgs = document.querySelectorAll("#output-image-upload img");

      if (imgs.nodeType === 1) {
          // just one image
        console.log("Found only a single node");
        imgList.push({
          id: imgs.id,
          index: 0,
          data: encodeURI(imgs.src).split(";base64,")[1],
        });
      } else {
        // It's an array of images
        var index = 0;
        imgs.forEach( img => {
          imgList.push({
            id: img.id,
            index: index,
            data: img.src.split(";base64,")[1],
          });
          index++;
        })
      }
      var model_type = $(
        '#dropdown-model-selection span[aria-selected="true"]'
      ).html();
      model_type=get_model_type(model_type)
      
        // Send the socket to the server
      socket.emit("image-upload", {
        model_type: model_type,
        images: imgList,
      });
      console.log("Socket emitted.");
      $('#server-toast').remove();
      $(document.body).append(`<div class="animate__animated animate__bounceInUp toast fade show shadow bg-info" role="alert" aria-live="assertive" aria-atomic="true" 
      data-autohide="false" style="position:fixed; bottom:3%; right:3%; min-width:350px;" id="server-toast">
      <div class="toast-header text-info" style="min-width-350px;">
        
        <strong class="mr-auto">Communication with server</strong>
        
        <button type="button" class="ml-2 mb-1 close" data-dismiss="toast" aria-label="Close">
          <span aria-hidden="true">×</span>
        </button>
      </div>
      <div class="toast-body bg-light">
        Sending image(s) to the server ...
      </div>
    </div>`)

      if( ! $('#output-image-process h3').length){
        $('#output-image-process').prepend('<h3 class="ml-5 text-primary font-weight-bold flex-break">Output</h3>')
      }
      $('#output-image-process .row.mt-5').remove();

      


    } else {
      // Output error showing that must type server URL.
      url_input.addClass(
        "border border-danger  animate__animated animate__shakeX"
      );
      setTimeout(
        () =>
          url_input.removeClass(
            "border border-danger  animate__animated animate__shakeX"
          ),
        10000
      );
    }
  }
};

const send_video_to_server=()=>{
  var url_input = $("#server-url-control input");
  if (url_input) {
    if (url_input.val()) {// If server URL is provided
        // If socket not initialized yet, connect and create handlers.
      if (socket === null || socket.disconnected|| socket.io.uri!==url_input.val()) {
        instantiate_socket(url_input.val())
      }

      console.log("VIDEOSocket emitted.");
      $('#server-toast').remove();
      $(document.body).append(`<div class="animate__animated animate__bounceInUp toast fade show shadow bg-info" role="alert" aria-live="assertive" aria-atomic="true" 
      data-autohide="false" style="position:fixed; bottom:3%; right:3%; min-width:350px;" id="server-toast">
      <div class="toast-header text-info" style="min-width-350px;">
        
        <strong class="mr-auto">Communication with server</strong>
        
        <button type="button" class="ml-2 mb-1 close" data-dismiss="toast" aria-label="Close">
          <span aria-hidden="true">×</span>
        </button>
      </div>
      <div class="toast-body bg-light">
        Sending image(s) to the server ...
      </div>
    </div>`)

  
     

    } else {
      // Output error showing that must type server URL.
      url_input.addClass(
        "border border-danger  animate__animated animate__shakeX"
      );
      setTimeout(
        () =>
          url_input.removeClass(
            "border border-danger  animate__animated animate__shakeX"
          ),
        10000
      );
    }
  }


}
// Socket handlers functions
function processImageResponse(data) {
  console.log("message received from server ", data);
  
  $('#server-toast .toast-body').html('Result image received.');
  $("#output-image-process").append(
    `<div class="row mt-5">
            <div class="col-md d-flex justify-content-center animate__animated animate__fadeInRight">
                <div class="d-flex flex-column align-items-center">
                    <h4 class="muted text-center"> Original </h4>
                    <img class="img-fluid" src="${decodeURI( "data:image/png;base64, "+imgList.find(el=>el.id===data["id"]).data)}">
                </div>
            </div>
            <div class="col-md d-flex justify-content-center animate__animated animate__fadeInRight">
                <div class="d-flex flex-column align-items-center">
                    <h4 class="muted text-center"> Estimated count : ${data["count"]} </h4>
                    <img class="img-fluid" src="${data["data"]}">
                </div>
            </div>
        </div>
        `
  );
}
  // Create an observer instance linked to the callback function
var observer = new MutationObserver(callback);

  // Start observing the target node for configured mutations
observer.observe(document.body, config);
  // Later, you can stop observing
// observer.disconnect();


document.ondrop = function (e) {
  if(!(e.currentTarget.id==='upload-image')){
    e.preventDefault();
    return;
  }
  console.log("dropped");
  console.log(e.target);
  $("#drop-div").addClass("d-none");
  let uploadDiv = document.getElementById("upload-image");
  console.log(uploadDiv)
  let loadingSpin = document.createElement("div");
  loadingSpin.innerHTML +='<div class="text-center align-self-center d-flex align-items-center justify-content-center" id="upload-loading"><div class="spinner-border text-primary" style="width: 7rem; height: 7rem;" role="status"><span class="sr-only">Loading...</span></div></div>';
  let a = document.createElement("div");
  a.id='upload-loading';
  a.className='text-center align-self-center d-flex align-items-center justify-content-center'
  let b = document.createElement("div");
  b.className='spinner-border text-primary'
  b.attributes.role='status'
  b.style.width='7em';
  b.style.height='7em';
  let c = document.createElement("div");
  b.appendChild(c);
  a.appendChild(b);
  loadingSpin.appendChild(a);
  uploadDiv.appendChild(loadingSpin);

  // Select the node that will be observed for mutations
  var targetNode = $("#output-image-upload");

  //   // Start observing the target node for configured mutations
     //observer.observe(document.body, config);
};
