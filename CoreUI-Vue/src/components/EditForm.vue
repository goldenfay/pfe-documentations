<template>
	<form @submit.prevent="onSubmit"  method="post">   
                <div class="form-group">
                  <label for="sensorName">Nom Capteur</label>
                  <input v-model="nameSensor" type="text" class="form-control" id="sensorName" placeholder="Entrer nom...">
                </div>

                <div class="form-group">
                  <label for="sensorZone">Nom de la zone</label>
                  <input v-model="nameZone" type="text" class="form-control" id="sensorZone" placeholder="Entrer nom de la zone...">
                </div>

                <div class="form-group">
                  <label for="sensorDescription">Description du capteur</label>
                  <textarea v-model="sensorDesc" class="form-control" id="sensorDescription" rows="3"></textarea>
                </div>

                
                <div class="form-group">
                  <label for="typeScene">Type de la scène</label>
                  <select  v-model="typeSc" class="form-control" id="typeScene" placeholder="Choisir type...">
                    <option v-for="type in types" v-bind:value="{ id: type.id, text: type.name }">{{ type.name }}
                    </option>
                  </select>
                </div>
                <Button type="submit" class="btn btn-primary  btn-block">Modifier capteur</Button>
              </form>
</template>

<script>

export default{
	name: 'EditForm',
	props:{
		e : {
			type: Object,
			required: true
		}
	},
	data(){
		return {
			      types : [{id:1,name:'Scène large échelle'},{id:2,name:'Scène moyen échelle'}],
			      nameSensor : null,
			      nameZone : null,
			      sensorDesc : null,
			      typeSc : '',

		}
	}
}

</script>