<template>
	<table class="table table-hover table-bordered table-striped">
     	<thead>
            <tr>
                  <th scope="col">ID</th>
                  <th scope="col">Nom Capteur</th>
                  <th scope="col">Nom Zone</th>
                  <th scope="col">Description</th>
                  <th scope="col">Type</th>
            </tr>
        </thead>
        <tbody>
                <tr v-for="sensor in allDBsensors">
                  <th scope="row" :id="sensor.sensor_name">{{sensor.id}}</th>
                  <td>{{sensor.sensor_name}}</td>
                  <td>{{sensor.sensor_zone}}</td>
                  <td>{{sensor.sensor_desc}}</td>
                  <td>{{sensor.sensor_type_name}}</td>
                </tr>
        </tbody>
    </table>
</template>


<script>


  export default {
    name: 'TableAffichage',
    props:{
    	allDBsensors:{
    		type : Array,
    		required : true
    	}
    },
  };
</script>