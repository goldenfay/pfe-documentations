<template>
	<div class="modal fade" :id="id" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">

		<div class="modal-dialog" role="document">
		  <div class="modal-content">

		      <div class="modal-header">
		        <slot name="title">
		        </slot>
		        <button type="button" class="close" data-dismiss="modal" aria-label="Close" @click="close">
		          <span aria-hidden="true">&times;</span>
		        </button>
		      </div>


		      <div class="modal-body">
		        <slot name="body_content"></slot>
		      </div>


		      <div class="modal-footer">
		        <slot name="button1"></slot>
		        <slot name="button2" data-dismiss="modal"></slot>
		      </div>

		  </div>
		</div>
	</div>
</template>

<script>
  export default {
    name: 'Modal',
    props:{
    	id:{
    		type : String,
    		required : true
    	}
    },
    methods: {
      close() {
        this.$emit('close');
      },
    },
  };
</script>