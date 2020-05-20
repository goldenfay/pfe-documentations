 // Options for the observer (which mutations to observe)
 const config = { attributes: true, childList: true, subtree: true };

 // Callback function to execute when mutations are observed
 const callback = function (mutationsList, observer) {
     // Use traditional 'for loops' for IE 11
     for (let mutation of mutationsList) {
         if (mutation.type === 'childList') {
            document.getElementById('upload-loading').remove()
            for( let addNode of mutation.addedNodes){
                if(addNode.id == "bar") {
                    console.log("bar was added!");
                }
            } 
            console.log('A child node has been added or removed.');
             console.log(mutation.target)

         }
         else if (mutation.type === 'attributes') {
             //console.log('The ' + mutation.attributeName + ' attribute was modified.');
         }
     }
 };

 // Create an observer instance linked to the callback function
 var observer = new MutationObserver(callback);

window.onload = function () {

    console.log('Helooooooooo')


    // Later, you can stop observing
    // observer.disconnect();

  



}

document.ondrop = function () {
    console.log('dropped')

    let uploadDiv = document.getElementById('upload-image');
    let loadingSpin = document.createElement('div')
    loadingSpin.innerHTML = '<div class="text-center align-self-center d-flex align-items-center justify-content-center" id="upload-loading"><div class="spinner-border text-primary" style="width: 7rem; height: 7rem;" role="status">\
    <span class="sr-only">Loading...</span></div></div>'
    uploadDiv.appendChild(loadingSpin)

    // Select the node that will be observed for mutations
    var targetNode = document.getElementById('output-image-upload');

    console.log(targetNode);
   

    // Start observing the target node for configured mutations
    observer.observe(targetNode, config);
    console.log(targetNode);


}