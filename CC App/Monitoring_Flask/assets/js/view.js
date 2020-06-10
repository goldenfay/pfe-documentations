console.log('heloooo')
var isAdvancedUpload = function() {
    var div = document.createElement('div');
    return (('draggable' in div) || ('ondragstart' in div && 'ondrop' in div)) && 'FormData' in window && 'FileReader' in window;
}();
$(window).load(function() {
    console.log('heloooo')
    var $form = $('.box');

    if (isAdvancedUpload) {
        console.log('yeees')

        $form.addClass('has-advanced-upload');
        var droppedFiles = false;
        var $input = $form.find('input[type="file"]'),
            $label = $form.find('label'),
            showFiles = function(files) {
                $label.text(files.length > 1 ? ($input.attr('data-multiple-caption') || '').replace('{count}', files.length) : files[0].name);
            };

        $form.on('drag dragstart dragend dragover dragenter dragleave drop', function(e) {
                e.preventDefault();
                e.stopPropagation();
            })
            .on('dragover dragenter', function() {
                $form.addClass('is-dragover');
            })
            .on('dragleave dragend drop', function() {
                $form.removeClass('is-dragover');
            })
            .on('drop', function(e) {
                droppedFiles = e.originalEvent.dataTransfer.files; // the files that were dropped
                showFiles(droppedFiles);
            });
        $form.on('submit', function(e) {
            if ($form.hasClass('is-uploading')) return false;

            $form.addClass('is-uploading').removeClass('is-error');

            if (isAdvancedUpload) {
                // ajax for modern browsers
                e.preventDefault();

                var ajaxData = new FormData($form.get(0));

                if (droppedFiles) {
                    $.each(droppedFiles, function(i, file) {
                        ajaxData.append($input.attr('name'), file);
                    });
                }
                $.ajax({
                    url: $form.attr('action'),
                    type: $form.attr('method'),
                    data: ajaxData,
                    dataType: 'json',
                    cache: false,
                    contentType: false,
                    processData: false,
                    complete: function() {
                        $form.removeClass('is-uploading');
                    },
                    success: function(data) {
                        $form.addClass(data.success == true ? 'is-success' : 'is-error');
                        if (!data.success) $errorMsg.text(data.error);
                    },
                    error: function() {
                        // Log the error, show an alert, whatever works for you
                    }
                });
            } else {
                // ajax for legacy browsers
                var iframeName = 'uploadiframe' + new Date().getTime();
                $iframe = $('<iframe name="' + iframeName + '" style="display: none;"></iframe>');

                $('body').append($iframe);
                $form.attr('target', iframeName);

                $iframe.one('load', function() {
                    var data = JSON.parse($iframe.contents().find('body').text());
                    $form
                        .removeClass('is-uploading')
                        .addClass(data.success == true ? 'is-success' : 'is-error')
                        .removeAttr('target');
                    if (!data.success) $errorMsg.text(data.error);
                    $form.removeAttr('target');
                    $iframe.remove();
                });
            }
        });
        $input.on('change', function(e) {
            showFiles(e.target.files);
        });
    }
});