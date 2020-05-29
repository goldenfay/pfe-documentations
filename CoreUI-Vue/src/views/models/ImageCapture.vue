<template>
  	<CCol sm="6" md="6">
      <CCard>

        <CCardHeader>
          <h1 class="display-6">L'image capturée par le capteur</h1>
        </CCardHeader>

        <CCardBody>
        	<video width="320" height="240" controls>
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
		name : 'ImageCapture',
		data(){
			return {imgUrl :require('@/../public/video/teb7iraVoyage.mp4')}


		},
		created(){
			this.$store.dispatch('addImage')

		},
		computed: mapState({
			name: state => state.test
		})

			/*getPath2(){
				return require('@/../public/video/'+this.$store.getters.getPath)
			}*/
		

	//end exporting	
	//@/../../../../hads/Downloads/
	}


</script>