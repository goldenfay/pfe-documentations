<template>
	<form @submit.prevent="onSubmitEdit(editFormData)"  method="post">   
    <div class="form-group">
      <label for="SN">Nom Capteur</label>
      <input v-model="editFormData.sensor_name" type="text" class="form-control" id="SN" placeholder="">
    </div>

    <div class="form-group">
      <label for="SZ">Nom de la zone</label>
      <input v-model="editFormData.sensor_zone" type="text" class="form-control" id="SZ" placeholder="Entrer nom de la zone...">
    </div>

    <div class="form-group">
      <label for="SD">Description du capteur</label>
      <textarea v-model="editFormData.sensor_desc" class="form-control" id="SD" rows="3"></textarea>
    </div>

                
    <div class="form-group">
      <label for="TS">Type de la scène</label>
      <select  v-model="typeSc" class="form-control" id="TS" placeholder="Choisir type...">
          <option v-for="type in types" v-bind:value="{ id: type.id, text: type.name }">{{ type.name }}
                    </option>
      </select>
    </div>

    <Button type="submit" class="btn btn-primary  btn-block">
      Modifier capteur
    </Button>
    <Button type="button" @click="cancel" class="btn btn-secondary btn-block">
      Annuler
    </Button>
  </form>
</template>

<script>

export default{
	name: 'EditForm',
	props:{
		editFormData:{
			type: Object,
			required: true
		}
	},
	data(){
		return {
			      types : [{id:1,name:'Scène large échelle'},{id:2,name:'Scène moyen échelle'}],
			      typeSc : '',

		}
	},
  methods:{
    onSubmitEdit(edited_data){
      console.log('Emit the edit event')
      edited_data.sensor_type_name = this.typeSc.text
      edited_data.sensor_type_id = this.typeSc.id
      this.$emit('edit-done',edited_data)

    },
    cancel(){
      this.$emit('cancel-edit')
    }
  },
  created(){

  }
}

</script>