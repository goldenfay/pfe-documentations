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
  return context
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
        
        

        var drawVLineCanvasBtn = $(addNode).find("#sensor-line-canvas-button");
        if (drawVLineCanvasBtn.length) {
          // if (addNode.id === "line-canvas-button") {
          console.log("captured");
          // Draw current scene on the canvas from the frame
          canvas = document.querySelector("#sensor-scene-canvas");
          // var canvas = document.createElement("canvas");
          var ctx = canvas.getContext("2d");
          var currentImg = document.querySelector("#sensor-process-video-output-flow");
          // Define Click handler
          $(drawVLineCanvasBtn).click(function (e) {
            $("#sensor-edit-canvas-area").removeClass("d-none");

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
                  point1 = {
                    x: e.clientX - canvas.getBoundingClientRect().x,
                    y: e.clientY - canvas.getBoundingClientRect().y,
                  };
                  return;
                } else {
                  if (point2 === null) {
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
                    // b=b/canvas.height;
                  }
                  console.log(tang,b)

                }
              },
              false
            );
          });
        }
          // Get reset canvas button
        var resetCanvasBtn = $(addNode).find("#sensor-clear-canvas-button");
        if (resetCanvasBtn.length) {
            // Add click event handler
          // if (addNode.id === "clear-canvas-button") {
          $(resetCanvasBtn).click((e) => {
            $("#sensor-edit-canvas-area").addClass("d-none");
            // ctx.clearRect(0, 0, currentImg.width, currentImg.height);
            ctx=clearCanvasShapes(ctx, currentImg);
          });
        }
            // Get Confirm canvas draw button
        var confirmDrawBtn = $(addNode).find("#sensor-confirm-draw-btn");
        if (confirmDrawBtn.length) {
            // Add click event handler
          // if (addNode.id === "confirm-draw-btn") {
          $(confirmDrawBtn).click((e) => {
            console.log(tang,b)
            $("#sensor-edit-canvas-area").addClass("d-none");
            $("#sensor-process-video-output-flow").remove();
            $('#sensor-edit-canvas-panel').hide();
            // $('#output-video-process').append('<p id="hidden-splitLine-input" class="d-none">'+tang+'/'+b+'</p>')
            // $('#hidden-splitLine-input').get(0).innerHTML=`${tang}/${b}`;
            $.post("http://localhost:8050/scene/regions/",
              {
                tang: tang,
                b: b,
                // p1: point1,
                // p2:point2
                p1: {'x': point1.x/canvas.width,'y':point1.y/canvas.height},
                p2: {'x': point2.x/canvas.width,'y':point2.y/canvas.height}
              },
              function(data, status){
                console.log('Received ', data, ' status', status);
                // location.reload();

            });
          });
        }
      }
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
      if (socket === null || socket.disconnected) {
        instantiate_socket(url_input.val())
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
      if( ! $('#output-image-process h3').length){
        $('#output-image-process').prepend('<h3 class="ml-5 text-primary font-weight-bold flex-break">Output</h3>')
      }
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


// Create an observer instance linked to the callback function
var observer = new MutationObserver(callback);


  // Start observing the target node for configured mutations
  observer.observe(document.body, config);
  // Later, you can stop observing
  // observer.disconnect();



