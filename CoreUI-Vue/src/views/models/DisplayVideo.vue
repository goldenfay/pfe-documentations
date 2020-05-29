<template>
  	<CCol sm="6" md="6">
      <CCard>
        <CCardHeader>
          <h1 class="display-6">La vidéo filmée par le capteur</h1>
        </CCardHeader>

        <CCardBody>
        	<video class="img-fluid" controls>
			  <source :src="require('@/../public/video/'+name)" type="video/mp4">
				Your browser does not support the video tag.
			</video> 
            <img src="" class="img-fluid"/>
        </CCardBody>

      </CCard>
    </CCol>
</template>	

<script>
	import CrowdServices from '@/services/CrowdServices.js'
	import {mapState} from 'vuex'

	export default{
		name : 'DisplayVideo',
		data(){
			return {imgUrl :require('@/../public/video/teb7iraVoyage.mp4')}


		},
		created(){
			this.$store.dispatch('getVideo')

		},
		computed: mapState({
			name: state => state.video_name
		})


	//end exporting	
	//@/../../../../hads/Downloads/
	}


</script>