<template>
	<CCol sm="6" md="6">
     <CCard>
      <CCardHeader>
        <h1 class="display-6">Informations du Comptage</h1>
      </CCardHeader>
      <CCardBody>
        <div class="embed-responsive embed-responsive-16by9">
        <iframe class="embed-responsive-item" src="http://127.0.0.1:8050/" allowfullscreen></iframe>
      </div>
      </CCardBody>
     </CCard>
    </CCol>
</template>


<script >
	export default{
		name: 'CountInfos'
	}
</script>