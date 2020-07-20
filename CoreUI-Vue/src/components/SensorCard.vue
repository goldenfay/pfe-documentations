<template>
  <router-link class="sensor-link" :to="{ name: 'list', params: { id: sensor.id } }">
    <div class="sensor-card -shadow">
      <span class="eyebrow">@12 on 1212</span>
      <h4 class="title "> {{sensor.sensor_name}} </h4>
      <BaseIcon name="users"> 25  attending</BaseIcon>
    </div>
  </router-link>
</template>

<script>
import BaseIcon from '@/components/BaseIcon.vue'

export default {
  name: 'SensorCard',
  components:{
    BaseIcon
  },
  props: {
    sensor: Object
  }
}
</script>

<style scoped>
    .sensor-card {
      padding: 20px;
      margin-bottom: 24px;
      transition: all 0.2s linear;
      cursor: pointer;
      background-color: white;
    }
    .sensor-card:hover {
      transform: scale(1.01);
      box-shadow: 0 3px 12px 0 rgba(0, 0, 0, 0.2), 0 1px 15px 0 rgba(0, 0, 0, 0.19);
    }
    .sensor-card > .title {
      margin: 0;
    }
    .sensor-link {
      color: black;
      text-decoration: none;
      font-weight: 100;
    }
</style>