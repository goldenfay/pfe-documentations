<template>
	<CCol sm="6" md="6">
		<CCard>
		<CRow><video ref="video" id="video" width="200" height="200" autoplay></video></CRow>
        <CRow><button id="snap" v-on:click="capture()">Snap Photo</button></CRow>
        <CRow>
        <canvas ref="canvas" id="canvas" width="200" height="200"></canvas>
        <ul>
            <li v-for="c in captures">
                <img v-bind:src="c" height="50" />
            </li>
        </ul>
    </CRow>
    </CCard>
	</CCol>		
</template>


<script >
	export default{
		name : 'VideoCapture',
		data(){
				return {video: {},
				                canvas: {},
				                captures: []}
		},
		mounted() {
		    this.video = this.$refs.video;
		    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
		        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
		            this.video.src = window.URL.createObjectURL(stream);
		            this.video.play();
		        });
		    }
		},
		methods: {
    		capture() {
		        this.canvas = this.$refs.canvas;
		        var context = this.canvas.getContext("2d").drawImage(this.video, 0, 0, 640, 480);
		        this.captures.push(canvas.toDataURL("image/png"));
    }
}

	//end exporting		
	}
</script>