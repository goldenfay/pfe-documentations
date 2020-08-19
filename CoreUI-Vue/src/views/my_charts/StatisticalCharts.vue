<template>
	<div>
		<CCol sm="12" md="12">
			  <CCard>
		        <CCardHeader>
		          Global Statistical Charts
		        </CCardHeader>
		        <CCardBody>
		            <div class="embed-responsive embed-responsive-1by1">
	        		<iframe class="embed-responsive-item" src="http://127.0.0.1:8050/sensors_dashboard" allowfullscreen></iframe>
	        		</div>
		        </CCardBody>
			  </CCard>	
		</CCol>	
		<!--CCol sm="12" md="12">
			<CCard>
		    <CCardBody>
		        <div class="embed-responsive embed-responsive-21by9">
	            <iframe class="embed-responsive-item" src="http://127.0.0.1:8050/mono" allowfullscreen>
	            </iframe>
	        	</div>
		    </CCardBody>
			</CCard>
		</CCol-->		
	</div>
</template>

<script >
	export default{
		name: 'StatisticalCharts'
	}
</script>