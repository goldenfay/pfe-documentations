<template>
	<div>
		 <CRow id="sensorTable">
      <CCol sm="12" md="12">
        <CCard>
          <CCardHeader>
            <h3>Mettre à jour capteur</h3>
          </CCardHeader>
          <CCardBody>
              <Table  :allDBsensors="allDBsensors" @delete-action="deleteSensorModal" @edit-action="editSensorModal" />

          </CCardBody>
          <CCardFooter>
            
          </CCardFooter>
        </CCard>

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
  name: 'UpdateSensor',
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


    editSensorModal(Sensor){
      var myModal = new cModal(document.getElementById('exampleModal'))
      this.editFormData = Object.assign({},Sensor)
      this.nameSensor = Sensor.sensor_name
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
      CrowdServices.DeleteRegistredSensor(this.deletedSensor)  
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
      CrowdServices.UpdateRegistredSensor(data,this.nameSensor,editedD.sensor_type_name)
      CrowdServices.postEditData(editedD.id,data).
        then(()=>{
          this.getMySensors()
          var myModalEl = document.getElementById('exampleModal')
          var myModal = cModal.getInstance(myModalEl)
          myModal.hide()  
        })
      console.log(this.nameSensor+' this is the name')
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
  }
}
</script>