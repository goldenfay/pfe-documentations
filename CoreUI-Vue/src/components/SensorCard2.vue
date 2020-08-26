<template>
  <!--router-link class="sensor-link" :to="{ name: 'list', params: { id: sensor.id } }"-->
    <div class="card">

      <div class="card-block">
        <slot name="card_image"></slot>
      </div>

      <div class="card-body text-center">
           <h5 class="card-title">{{sensor.sensor_name}}</h5>
           <h6 class="card-title">{{sensor.sensor_zone}}</h6>
            <p class="card-text">
              <br/>
              {{sensor.sensor_desc}}
            </p>

              <BaseIcon v-if="sensor.sensor_type_id==2" name="users" width=20 height=20>
              Scène à moyenne échelle
          	  </BaseIcon>
          	  <BaseIcon v-else name="users" width=20 height=20>
              Scène à grande échelle
          	  </BaseIcon>
      </div>

      <div class="card-body">
          <slot name="process_button">
            <router-link :to="{name:'ProcessSensor', params:{id:sensor.id,name:sensor.sensor_name,type:sensor.sensor_type_id }}" v-slot="{ href, route, navigate}">
                <button :href="href" @click="navigate" class="btn btn-primary btn-block" style="text" >
                  Commencer le traitement 
                </button>
                <!--ui-button-->
            </router-link>
          </slot>

          <slot name="edit_button">
            <router-link :to="{name:'SensorExstat', params:{name:sensor.sensor_name}}" v-slot="{ href, route, navigate}">
                <button :href="href" @click="navigate" class="btn btn-secondary btn-block" style="text" >
                  Plus d'informations
                </button>
            </router-link>
          </slot>
      </div>

    </div>
  <!--/router-link-->
</template>

<script>
import BaseIcon from '@/components/BaseIcon.vue'

export default {
  name: 'SensorCard2',
  components:{
    BaseIcon
  },
  props: {
    sensor: Object
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
    .btn-secondary:hover{
      background-color: rgba(239, 108, 0, 0.8);
      border-color: rgba(239, 108, 0, 0.8); 
      color: white;
    }
</style>