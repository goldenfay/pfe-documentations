<template>
  <div>
    <CRow>
     <!-- <DisplayVideo/>
      <CountInfos/> -->
      <CCol sm="6" md="4">
      <div class="shadow p-3 mb-5 bg-white rounded">
         
            <!--CCol md="1">
              <img src="@/assets/icons/flask.png">
            </CCol>
            <CCol md="1">
              <img src="@/assets/icons/pycv.png">
            </CCol-->
            <div class="card">
              <div class="card-body">
                
                <strong>  Poste de travail N°1:</strong><span style="visibility: hidden;">Poste pp</span>
                <img src="https://img.icons8.com/color/64/000000/laptop--v1.png"/>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>GNU/Linux Ubuntu 18.0464bits</strong>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Mémoire vive 4GO</strong>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Processeur Intel i3-6200U CPU @2.30GHz</strong>
              </div>
            </div>


      </div>
    </CCol>
    <CCol sm="6" md="4">
      <div class="shadow p-3 mb-5 bg-white rounded">
         
            <!--CCol md="1">
              <img src="@/assets/icons/flask.png">
            </CCol>
            <CCol md="1">
              <img src="@/assets/icons/pycv.png">
            </CCol-->
            <div class="card">
              <div class="card-body">
                <strong>  Poste de travail N°2:</strong><span style="visibility: hidden;">Poste pp</span>
                <img src="https://img.icons8.com/color/64/000000/laptop--v1.png"/>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Windows 10 64bits</strong>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Mémoire vive 8GO</strong>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Processeur Intel i5-8565U CPU @4.60GHz</strong>
              </div>
            </div>

            
      </div>
    </CCol>
    <CCol sm="6" md="4">
      <div class="shadow p-3 mb-5 bg-white rounded">
         
            <!--CCol md="1">
              <img src="@/assets/icons/flask.png">
            </CCol>
            <CCol md="1">
              <img src="@/assets/icons/pycv.png">
            </CCol-->
            <div class="card">
              <div class="card-body">
                <strong>  Serveur Colab:</strong><span style="visibility: hidden;">Poste du travail  </span>
                <img src="https://img.icons8.com/dusk/64/000000/server.png"/>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Tesla K80 12 GBDDR5 VRAM</strong>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Mémoire vive 13GO</strong>
              </div>
            </div>
            <div class="card">
              <div class="card-body">
                <strong>Processeur Intel(R) Xeon(R) CPU @2.30GHz</strong>
              </div>
            </div>


      </div>
    </CCol>


    </CRow>

    <CRow>  
      <ImageDensity/>
    </CRow> 

  </div>
</template>

<script>

import DisplayVideo from '@/views/models/DisplayVideo'
import ImageDensity from '@/views/models/ImageDensity' 
import CountInfos from '@/views/models/CountInfos' 

export default {

    name: 'ModelShow',
    components:{
      DisplayVideo,
      ImageDensity,
      CountInfos
    },
    created(){
      this.$store.dispatch('getVideos')

    },
}
</script>
