<template>
	<div>
		<CCol sm="12" md="12">
			  <CCard>
		        <CCardHeader>

		          Sensor Statistics  

		        </CCardHeader>
		        <CCardBody>
		            <div class="embed-responsive embed-responsive-1by1">
	        		<iframe class="embed-responsive-item" :src="adress" allowfullscreen></iframe>
	        		</div>
		        </CCardBody>
			  </CCard>	
		</CCol>	
	</div>
</template>

<script>
//#http://localhost:8050/fr/statistics?sensor_name=sensor_1
export default {
	
  name: 'SensorExstat',
  data(){
  	return {
  		adress : "",
      sensor_name:""
  	}
  },
  created(){
    this.sensor_name = this.$route.params.name
    this.adress = "http://localhost:8050/fr/statistics?sensor_name="+this.sensor_name
  },
  methods:{
  },

  }

</script>