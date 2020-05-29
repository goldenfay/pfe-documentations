<template>
	<CCol sm="6" md="6">
     <CCard>
      <CCardHeader><h1 class="display-6">Map density</h1></CCardHeader>
        <CCardBody>
            <p class="lead">Insertion de la density map ici</p>
            <img :src="imgUrl" class="img-fluid">
        </CCardBody>
     </CCard>
    </CCol>
</template>


<script >
	export default{
		name: 'ImageDensity',
		data(){
			return {imgUrl : require('/home/hads/Downloads/density.png')}

		}
	}
</script>