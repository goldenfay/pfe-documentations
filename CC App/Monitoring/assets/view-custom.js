 // Options for the observer (which mutations to observe)
 const config = { attributes: true, childList: true, subtree: true };

 // Callback function to execute when mutations are observed
 const callback = function (mutationsList, observer) {
     // Use traditional 'for loops' for IE 11
     for (let mutation of mutationsList) {
         if (mutation.type === 'childList') {
            document.getElementById('upload-loading').remove()
            document.getElementById('drop-div').classList.remove('d-none')
            for( let addNode of mutation.addedNodes){
                console.log(addNode.id)
                if(addNode.id == "process-imgs-button") {
                    document.getElementById('process-imgs-button').onclick=function(e){
                        console.log(e.target.nodeName)
                    }
                }
            } 
            
         }
         else if (mutation.type === 'attributes') {
             //console.log('The ' + mutation.attributeName + ' attribute was modified.');
         }
     }
 };

 // Create an observer instance linked to the callback function
 var observer = new MutationObserver(callback);

window.onload = function () {

    console.log('Loading script ....')

   
    // Later, you can stop observing
    // observer.disconnect();

  



}

document.ondrop = function () {
    console.log('dropped')
    document.getElementById('drop-div').classList.add('d-none')
    let uploadDiv = document.getElementById('upload-image');
    let loadingSpin = document.createElement('div')
    loadingSpin.innerHTML = '<div class="text-center align-self-center d-flex align-items-center justify-content-center" id="upload-loading"><div class="spinner-border text-primary" style="width: 7rem; height: 7rem;" role="status">\
    <span class="sr-only">Loading...</span></div></div>'
    uploadDiv.appendChild(loadingSpin)

    // Select the node that will be observed for mutations
    var targetNode = document.getElementById('output-image-upload'); 

    // Start observing the target node for configured mutations
    observer.observe(targetNode, config);
    


}