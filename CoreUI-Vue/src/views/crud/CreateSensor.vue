<template>
	<div>
		 <CRow>
      <CCol sm="12" md="12">
        <CCard>
          <CCardHeader>
            <h3>Création d'un capteur</h3>
          </CCardHeader>
          <CCardBody>
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
                <Button type="submit" class="btn btn-primary  btn-block">Créer capteur</Button>
              </form>
          </CCardBody>
          <CCardFooter>
            
          </CCardFooter>
        </CCard>

        <!-- Button trigger modal --
        <button type="button" class="btn btn-primary" @click="boolModal1 = true">
          Launch demo modal
        </button>-->
        <Modal  id="exampleModal"  @close="boolModal1 = false">
              
                <h2 slot="title">Modification Capteur</h2>
                <EditForm slot="body_content" :editFormData="editFormData" @edit-done="postEditedData" @cancel-edit="close_edit"/>
        </Modal>

        <Modal id="deleteModal">
            <h1 slot="title">Confirmer la suppression</h1>
            <p slot="body_content">Êtes vous sur de vouloir supprimer ?</p>
            <button @click="deleteSensor" slot="button1" type="button" class="btn btn-danger" >Supprimer</button>
            <button slot="button2" type="button" class="btn btn-secondary" data-dismiss="modal">Annuler</button>
        </Modal>

        <Modal id="createModal">
            <div slot="body_content" class="alert alert-success" role="alert">
              <h2> Capteur créer avec succès !</h2> 
            </div>
            <button slot="button2" type="button" class="btn btn-secondary" data-dismiss="modal">Fermer</button>
        </Modal>  
      </CCol>

    </CRow>
    <CRow id="sensorTable">
      <CCol sm="12" md="12">
        <CCard>
          <CCardBody >
              <Table  :allDBsensors="allDBsensors" @delete-action="deleteSensorModal" @edit-action="editSensorModal" />
          </CCardBody>  
        </CCard>    
      </CCol>
    </CRow>

	</div>
</template>

<script>

import CrowdServices from '@/services/CrowdServices.js'
import Modal from '@/components/Modal'
import Table from '@/components/Table'
import EditForm from '@/components/EditForm'
import ActionButton from '@/components/ActionButton'
import apiClient from '@/services/CrowdServices.js'
import $ from 'jquery'
import {Toast,Modal as cModal} from '@coreui/coreui'

export default {
  name: 'CreateSensor',
  components:{
    Modal,
    Table,
    ActionButton,
    EditForm
  },
  data(){
    return {
      boolModal1: false,
      valr: null,
      nameSensor : null,
      nameZone : null,
      sensorDesc : null,
      typeSc : '',
      types : [{id:1,name:'Scène large échelle'},{id:2,name:'Scène moyen échelle'}],
      allDBsensors: [],
      editFormData: {},
      deletedSensor: {}
    }
  },
  methods:{

    onSubmit(){
      var data = {'sensor_name': this.nameSensor,
              'sensor_zone': this.nameZone,
              'sensor_desc': this.sensorDesc,
              'sensor_type_name': this.typeSc.text,
              'sensor_type_id': this.typeSc.id
      }
      CrowdServices.postSensorData(data).
        then(()=>{ 
            var myModal = new cModal(document.getElementById('createModal'))
            myModal.show()
            this.getMySensors()
        })
      CrowdServices.registerSensor(this.nameSensor,this.typeSc.text)
      this.nameSensor = null;
      this.nameZone = null;
      this.sensorDesc = null;
      this.typeSc = '';
    },

    editSensorModal(Sensor){
      var myModal = new cModal(document.getElementById('exampleModal'))
      this.editFormData = Object.assign({},Sensor)
      myModal.show()
    },

    deleteSensorModal(Sensor){
      var myModal = new cModal(document.getElementById('deleteModal'))
      this.deletedSensor = Object.assign({},Sensor)
      myModal.show()

    },

    deleteSensor(){
      CrowdServices.deleteSensorRequest(this.deletedSensor.id).
        then(()=>{
            this.getMySensors()
            var myModalEl = document.getElementById('deleteModal')
            var myModal = cModal.getInstance(myModalEl)
            myModal.hide()  
        })
      //var myModal = new cModal(document.getElementById('deleteModal'))
      //var myModal = document.getElementById('deleteModal')  
    },

    postEditedData(editedD){
      var data = {'sensor_name': editedD.sensor_name,
              'sensor_zone': editedD.sensor_zone,
              'sensor_desc': editedD.sensor_desc,
              'sensor_type_name': editedD.sensor_type_name,
              'sensor_type_id': editedD.sensor_type_id
      }
      CrowdServices.postEditData(editedD.id,data).
        then(()=>{
          this.getMySensors()
          var myModalEl = document.getElementById('exampleModal')
          var myModal = cModal.getInstance(myModalEl)
          myModal.hide()  
        })
      editFormData: {}
    },

    getMySensors(){
        CrowdServices.getSensors()
        .then((response)=>{
          this.allDBsensors = response.data
          //console.log(this.allDBsensors[0].sensor_name)
        })
        .catch(error =>{
          console.log(error)
        }) 
    },

    close_edit(){
      var myModal = new cModal(document.getElementById('exampleModal'))
      console.log('wsalt')
      //myModal.hide()
      console.log('wsalt2')
    }

  },
  created(){
    this.getMySensors() 
  },
  computed:{
    test(){
      return this.typeSc.text+" "+this.nameSensor+" "+this.nameZone
    }
  }
}
</script>