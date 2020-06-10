 // Options for the observer (which mutations to observe)
 const config = { attributes: true, childList: true, subtree: true };

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
         if (url_input.val()) {

             console.log('Value : ', url_input.val())
             var imgs = document.querySelector('#output-image-upload img')
             if (imgs.nodeType === 1) { // just one image
                 console.log('It is a node')
                 var socket = io.connect(url_input.val(), {
                     reconnection: false
                 });
                 //socket.eio.pingTimeout = 180000;
                 socket.emit('image-upload', {
                     id: imgs.id,
                     index: 0,
                     data: imgs.src
                 });
                 socket.on('connect',
                     function() {
                         console.log('initSocketIO')
                     });
                 socket.on("send-image", processImageResponse);
             } else { // It's an array of images

                 for (let img in imgs) {
                     console.log('Image length :', img)

                 }
             }
         } else {
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