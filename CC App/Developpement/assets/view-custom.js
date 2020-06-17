 // Options for the observer (which mutations to observe)
 const config = { attributes: true, childList: true, subtree: true };
 var socket = null

 const error_alert = function(errorTitle, errorMessage) {
         return `<div class="error-container">
            <div class="alert alert-danger" role="alert"  data-toggle="collapse" data-target="#collapseMessage" aria-expanded="false" aria-controls="collapseMessage">
            ${errorTitle}
            </div>
            <div class="card card-body" id ="collapseMessage">
                <span class="fa fa-times-circle text-danger"></span> ${errorMessage}  
            </div>
            </div>
        
        `
     }
     // Callback function to execute when mutations are observed
 const callback = function(mutationsList, observer) {
     // Use traditional 'for loops' for IE 11
     for (let mutation of mutationsList) {
         if (mutation.type === 'childList') {
             if ($('#upload-loading'))
                 $('#upload-loading').remove()
             $('#drop-div').removeClass('d-none')
             for (let addNode of mutation.addedNodes) {
                 //  console.log(addNode)
                 var processbtn = $(addNode).find('#process-imgs-button')
                 if (processbtn) {
                     console.log('Button targeted')

                     processbtn.click(function(e) {
                         if ($('#usage-switch').hasClass('toggled-on')) {
                             console.log('Clicked')
                             send_to_server();
                         }

                     });
                 }
             }

         } else if (mutation.type === 'attributes') {
             //console.log('The ' + mutation.attributeName + ' attribute was modified.');
         }
     }
 };
 const send_to_server = function() {
     var url_input = $('#server-url-control input')
     if (url_input) {
         if (url_input.val()) { // If server URL is provided
             var imgList = Array();
             // If socket not initialized yet, connect and create handlers.
             if (socket === null) {
                 socket = io.connect(url_input.val(), {
                     reconnection: false
                 });
                 socket.on('connect',
                     function() {
                         console.log('Socket connected.');
                     });
                 socket.on('server-error',
                     function(data) {
                         $(document).append(error_alert('An error occured on the server.', data['message']));
                         this.disconnect();

                     });
                 socket.on('process-done',
                     function(data) {
                         errors = data['errors']
                         if (errors && errors[0]) {

                             $(document).append(error_alert('An error occured on the server.', errors))
                         }

                     });
                 socket.on("send-image", processImageResponse);
             }
             if (!socket.connected) {
                 socket = io.connect(url_input.val(), {
                     reconnection: false
                 });

             }


             var imgs = document.querySelector('#output-image-upload img')

             if (imgs.nodeType === 1) { // just one image
                 console.log('Found only a single node')
                 imgList.push({
                     id: imgs.id,
                     index: 0,
                     data: encodeURI(imgs.src).split(";base64,")[1]
                 });

             } else { // It's an array of images
                 var index = 0;
                 for (let img in imgs) {
                     console.log('Image length :', img)
                     imgList.push({
                         id: img.id,
                         index: index,
                         //  data: unescape(encodeURIComponent(img.src)).split(";base64,")[1]
                         data: encodeURI(img.src).split(";base64,")[1]
                     });
                     index++;
                 }
             }
             var model_type = $('#dropdown-model-selection span[aria-selected="true"]').html()
             switch (model_type) {

                 case ('Mobile SSD'):
                     model_type = 'mobileSSD';
                     break;
                 case ('YOLO'):
                     model_type = 'yolo';
                     break;
                 case ('MCNN'):
                     model_type = 'MCNN';
                     break;
                 case ('CSRNet'):
                     model_type = 'CSRNet';
                     break;
                 case ('SANet'):
                     model_type = 'SANet';
                     break;
                 case ('CCNN'):
                     model_type = 'CCNN';
                     break;
             }
             console.log(model_type)
             socket.emit('image-upload', {
                 model_type: model_type,
                 images: imgList
             });

         } else { // Output error showing that must type server URL.
             url_input.addClass('border border-danger  animate__animated animate__shakeX')
             setTimeout(() => url_input.removeClass('border border-danger  animate__animated animate__shakeX'), 10000)

         }
     }
 }

 // Socket handlers functions
 function processImageResponse(data) {
     $('<div/>', {
             class: 'row'
         })
         .append(
             $('<div/>', {
                 class: 'col-md justify-content-center animate__animated animate__fadeInRight'
             }).append(
                 $('<div/>', {
                     class: 'd-flex justify-content-center'
                 }).append(

                     $('<h4/>', {
                         text: 'Original',
                         class: 'muted'
                     })
                 )
             ).append(
                 $('<img/>', {
                     id: 'img',
                     src: data
                 })

             )
         )
         .appendTo('#output-image-process');

 }
 // Create an observer instance linked to the callback function
 var observer = new MutationObserver(callback);

 window.onload = function() {


     // $('#usage-switch').attrchange({
     //     trackValues: true, 
     //     callback: function (event) {
     //         if(evnt.attributeName == "class") { 
     //             if(evnt.newValue.search('toggled-on') >0) { // "open" is the class name you search for inside "class" attribute

     //                 var 
     //             }
     //         }
     //     }
     //   });


     // Later, you can stop observing
     // observer.disconnect();





 }

 document.ondrop = function() {
     console.log('dropped')
     $('#drop-div').addClass('d-none')
     let uploadDiv = $('#upload-image');
     let loadingSpin = document.createElement('div')
     loadingSpin.innerHTML = '<div class="text-center align-self-center d-flex align-items-center justify-content-center" id="upload-loading"><div class="spinner-border text-primary" style="width: 7rem; height: 7rem;" role="status">\
    <span class="sr-only">Loading...</span></div></div>'
     uploadDiv.append(loadingSpin)

     // Select the node that will be observed for mutations
     var targetNode = $('#output-image-upload');

     // Start observing the target node for configured mutations
     observer.observe(targetNode.get(0), config);



 }