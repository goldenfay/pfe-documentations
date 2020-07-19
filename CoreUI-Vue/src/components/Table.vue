<template>
	<table class="table table-hover table-bordered table-striped">
     	<thead>
            <tr>
                  <th scope="col">ID</th>
                  <th scope="col">Nom Capteur</th>
                  <th scope="col">Nom Zone</th>
                  <th scope="col">Description</th>
                  <th scope="col">Type</th>
                  <th scope="col">Action</th>
            </tr>
        </thead>
        <tbody>
                <tr v-for="sensor in allDBsensors">
                  <th scope="row">{{sensor.id}}</th>
                  <td>{{sensor.sensor_name}}</td>
                  <td>{{sensor.sensor_zone}}</td>
                  <td>{{sensor.sensor_desc}}</td>
                  <td>{{sensor.sensor_type_name}}</td>
                  <td>
                        <slot name="EditButton">
                              <Button :sensorsID="sensor.id" class="btn btn-success btn-sm" @click="fireEdit(sensor.id)">
                                  Modifier
                              </Button>
                        </slot>
                        <slot name="DeleteButton">
                              <Button :sensorsID="sensor.id" class="btn btn-danger btn-sm" @click="fireDelete(sensor.id)">
                                  Supprimer
                              </Button>
                        </slot>
                  </td>
                </tr>
        </tbody>
    </table>
</template>


<script>

import ActionButton from '@/components/ActionButton'

  export default {
    name: 'Table',
    components:{
      ActionButton
    },
    props:{
    	allDBsensors:{
    		type : Array,
    		required : true
    	}
    },
    methods:{
      fireDelete(id_Of_Sensor){
        this.$emit('delete-action',id_Of_Sensor)
      },
      fireEdit(id_Of_Sensor){
        this.$emit('edit-action',id_Of_Sensor)
      }
    }
  };
</script>