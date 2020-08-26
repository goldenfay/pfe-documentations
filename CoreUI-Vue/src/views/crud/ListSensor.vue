<template>
  <div>   
    <!--CRow>
      <CCol  sm="6" md="6">
          	    <h1 >Sensors Listing</h1>
          	    <button class="btn btn-rounded" type="text" :data-tooltip="highlightLinksText">
          	    	YOP
          	    </button>
          
          		<SensorCard v-for="sensor in sensors" :key="sensor.id" :sensor="sensor"/>
          		<template v-if="page != 1">
			      <router-link :to="{ name: 'ListSensor', query: { page: page - 1 } }" rel="prev">
			      Prev Page</router-link>
			    <template v-if="hasNextPage"> | </template>
			    </template>
			    <router-link v-if="hasNextPage" :to="{ name: 'ListSensor', query: { page: page + 1 } }" rel="next">
			      Next Page</router-link>
      </CCol-->
  <!--CCol sm="4" md="4">   
      <div class="card">
  <div class="card-body text-center">
    <h5 class="card-title">Card title</h5>
    <p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
  </div>
  <div class="card-body">
    <div class="container-fluid">
      <div class="row">
          <div class="col-md-6">
              <button class="btn btn-primary">Start </button>
          </div>
          <div class="col-md-6">
              <button class="btn btn-primary">Start </button>
          </div>
      </div>
    </div>
  </div>


</div>
</CCol>
 <CCol sm="4" md="4">   
      <div class="card">
  <div class="card-block">
      <img src="@/../public/img/avatars/crowd.jpg" class="card-img-top" alt="...">

  </div>
  <div class="card-body text-center">
    <font-awesome-icon icon="coffee" />
       <h5 class="card-title">Rue Didouche Morade</h5>
    <p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.
      <br/>

            <BaseIcon name="users" width=20 height=20> 25  attending</BaseIcon>
    </p>
  </div>
  <div class="card-body">
      <button type="button" class="btn btn-primary btn-block">Commencer le Traitement</button>

      <button type="button" class="btn btn-secondary btn-block">Block level button</button>
  </div>
</div>
</CCol-->

    <CRow >
        <CCol sm="4" md="4" v-for="sensor in sensors" :key="sensor.id">
          <SensorCard2 :key="sensor.id" :sensor="sensor">
              <img v-if="sensor.sensor_type_id == 2" slot="card_image" src="@/../public/img/avatars/crowd0.jpg" class="card-img-top" alt="...">
              <img v-else slot="card_image" src="@/../public/img/avatars/crowd3.jpg" class="card-img-top" alt="...">
          </SensorCard2>
        </CCol>  
    </CRow>    
    <div class="row justify-content-md-center">  	
      <div class="btn-toolbar" role="toolbar" aria-label="Toolbar with button groups">
        <div class="btn-group mr-2" role="group" aria-label="First group">

            <template v-if="page != 1">
              <router-link :to="{ name: 'ListSensor', query: { page: page - 1 } }" rel="prev" v-slot="{ href, route, navigate}">
                <button :href="href" @click="navigate" class="btn btn-secondary" style="text" >
                    Page précédente
                </button>    
              </router-link>
              <!--template v-if="hasNextPage"> | </template-->
            </template>
            
            <router-link v-for="n in nbPages" :to="{ name: 'ListSensor', query: { page: n } }" rel="next" v-slot="{ href, route, navigate}" :key="n">
                  <button  :id="n" :href="href" @click="navigate" class="btn btn-secondary" style="text"  checked>{{n}}</button>
            </router-link>
    
            <router-link v-if="hasNextPage" :to="{ name: 'ListSensor', query: { page: page + 1 } }" rel="next" v-slot="{ href, route, navigate}">
              <button :href="href" @click="navigate" class="btn btn-secondary" style="text" >
                  Page suivante
              </button>
            </router-link>
        </div> 
      </div>
    </div>  
    </br>


  </div>
</template>

<script>
import SensorCard from '@/components/SensorCard.vue'
import BaseIcon from '@/components/BaseIcon.vue'
import SensorCard2 from '@/components/SensorCard2.vue'
import Modal from '@/components/Modal'
import EditForm from '@/components/EditForm'
import { mapState } from 'vuex'

export default {
  name: 'ListSensor',
  components:{
    Modal,
  },  
  data(){
    return {
          URL : 2,
      }
  },
  components: {
    SensorCard,
    BaseIcon,
    SensorCard2
  },
  props:{
  	highlightLinksText: {
      type: String,
      default: "Highlight Links"
    }
  },
  mounted(){
      if( parseInt(this.$route.query.page) >= 1){    
          var nb_current_page = parseInt(this.$route.query.page)
          document.getElementById(nb_current_page).style.color = "white";
          document.getElementById(nb_current_page).style.backgroundColor = "blue";
      }
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
    nbPages(){
      if(this.sensorsTotal%3 == 0 ){
        return Math.floor(this.sensorsTotal/3)
      }
      else{
        return Math.floor(this.sensorsTotal/3)+1
      }
    },
    ...mapState(['sensors', 'sensorsTotal'])
  }
}
</script>

<style scoped>
  .card:hover {
  transform: scale(1.04);
  box-shadow: 5px 5px 15px rgba(0,0,0,0.6);
  }
  .card{
   border-radius: 18px;
  }
  .btn-primary:hover{
    background-color:green;
    border-color: green;
  }
</style>  

