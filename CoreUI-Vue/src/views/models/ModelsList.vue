<template>
  <div>
    <!--
    <CRow>
      <CCol sm="6" md="4">
        <CCard>
          <CCardHeader>
            <h1 class="display-6">MCNN model</h1>
          </CCardHeader>
          <CCardBody>
              <p class="lead">
                Multi Column Convolutional Neural Network </br>
                <span style="visibility: hidden;">Convolutional Neural Network cdcd</span>
              </p>

          </CCardBody>
          <CCardFooter><router-link to="/dashboard">Use Model</router-link></CCardFooter>
        </CCard>
      </CCol>
      <CCol sm="6" md="4">
        <CCard>
              <CCardHeader>
                <h1 class="display-6">CSR model</h1>
              </CCardHeader>
          <CCardBody>    
              <p class="lead">CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes</p>
          </CCardBody>
          <CCardFooter><router-link to="/dashboard">Use Model</router-link></CCardFooter>
        </CCard>
      </CCol>
      <CCol sm="6" md="4">
        <CCard>
          <CCardHeader><h1 class="display-6">SATNet model</h1></CCardHeader>
          <CCardBody>
              <p class="lead">SATNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes</p>
          </CCardBody>
          <CCardFooter><router-link to="/dashboard">Use Model</router-link></CCardFooter>
        </CCard>
      </CCol>
    </CRow>
  -->
  <CCRow>
    <CCol sm="12" md="12">
      <div class="embed-responsive embed-responsive-4by3">
              <iframe class="embed-responsive-item" src="http://127.0.0.1:8050/view#left-side-column" allowfullscreen>
              </iframe>
            </div>
    </CCol>
  </CCRow>
  </div>
</template>

<script>

export default {
  name: 'ModelsList'
}
</script>
