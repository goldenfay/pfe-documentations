<template>
  <div>   
    <CRow>
      <CCol sm="12" md="12">
          	    <h1>Sensors Listing</h1>
          
          		<SensorCard v-for="sensor in sensors" :key="sensor.id" :sensor="sensor"/>
          		<template v-if="page != 1">
			      <router-link :to="{ name: 'ListSensor', query: { page: page - 1 } }" rel="prev">
			      Prev Page</router-link>
			    <template v-if="hasNextPage"> | </template>
			    </template>
			    <router-link v-if="hasNextPage" :to="{ name: 'ListSensor', query: { page: page + 1 } }" rel="next">
			      Next Page</router-link>
      </CCol>
    </CRow>      	
  </div>
</template>

<script>
import SensorCard from '@/components/SensorCard.vue'
import { mapState } from 'vuex'

export default {
  components: {
    SensorCard
  },
  created() {
  	this.$store.dispatch('fetchSensors',{
  		perPage: 3,
  		page: this.page
  	})
  },
  computed:{
  	page(){
  		return parseInt(this.$route.query.page) || 1
  	},
  	hasNextPage() {
      return this.sensorsTotal > this.page * 3
    },
    ...mapState(['sensors', 'sensorsTotal'])
  }
}
</script>