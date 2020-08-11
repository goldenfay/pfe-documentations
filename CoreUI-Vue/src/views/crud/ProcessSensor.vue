<template>
	<div>
		<CCol sm="12" md="12">
			  <CCard>
		        <CCardHeader>
		          Sensor Processing {{test()}}
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
//this.server_url+'/process'+'?sensor_name='+sensor.name+'&sensor_id='+sensor.id+'&model_name=sensor.type'
export default {
	
  name: 'ProcessSensor',
  data(){
  	return {
  		adress : "",
  		name_sensor:"",
  		type_sensor:"",
  		id_sensor: null,
  		id_type_sensor: null
  	}
  },
  created(){
  	this.adress = "http://localhost:3000/crowds"
  	this.id_sensor = this.$route.params.id
  	this.name_sensor = this.$route.params.name
  	this.id_type_sensor = this.$route.params.type
  	if ( this.id_type_sensor == 1 ){
  		this.type_sensor = "SANet"
  	}
  	else{
  		this.type_sensor = "ssd"
  	}

  },
  methods:{
  	test(){
  		return this.type_sensor
  	}
  },

  }

</script>