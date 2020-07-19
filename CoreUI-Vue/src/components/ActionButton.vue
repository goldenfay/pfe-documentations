<template>
	<button type="button" class="btn btn-primary">
		<slot/>
	</button>
</template>

<script>
  export default {
    name: 'ActionButton',
    props:{
    	sensorsID: Number,
    	required: true
    }
  };
</script>