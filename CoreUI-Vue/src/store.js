import Vue from 'vue'
import Vuex from 'vuex'
import CrowdServices from '@/services/CrowdServices.js'
import axios from 'axios'


Vue.use(Vuex)

export default new Vuex.Store({

  state:{
        sensors: [],
        sensorsTotal: 0,
        test : '',
        video_name : 'teb7iraVoyage',
        allImages : [],
        all_videos : [],
        //Toggle state 
        sidebarShow: 'responsive',
        sidebarMinimize: false    
  },
  mutations:{

        SET_SENSORS_TOTAL(state, sensorsTotal) {
          state.sensorsTotal = sensorsTotal
         },
        SET_SENSORS(state,sensors){
            state.sensors = sensors
        },
        ADD_IMAGE(state,images){
            state.test = images[0].title
            console.log('rani dfel mutations',images[0].title)
        },
        ADD_VIDEO(state,video){
            state.video_name = video[0].title
        },
        ADD_VIDEOS(state,videos){
            for(let i=0;i<videos.length;i++){
              console.log('This is it: ',videos[i].title)
            }
        },

        //Toggle MUTATIONS
        toggleSidebarDesktop (state) {
          const sidebarOpened = [true, 'responsive'].includes(state.sidebarShow)
          state.sidebarShow = sidebarOpened ? false : 'responsive'
        },
        toggleSidebarMobile (state) {
          const sidebarClosed = [false, 'responsive'].includes(state.sidebarShow)
          state.sidebarShow = sidebarClosed ? true : 'responsive'
        },
        set (state, [variable, value]) {
          state[variable] = value
        }
  },
  actions:{
        fetchSensors({commit},{perPage, page}){
          CrowdServices.getSensorsPagination(perPage,page)
            .then(response => {
             commit(
                'SET_SENSORS_TOTAL',
                parseInt(response.headers['x-total-count'])
              )
              commit('SET_SENSORS',response.data)
            })
            .catch(error => {
              console.log('There was an error:', error.response)
            })
        },
        addImage({commit}){
          /*axios.get('http://localhost:3000/events')
          .then(response => {

                commit('ADD_IMAGE',response.data)

          })
          .catch(error =>{
            console.log('Un message d erreur : ',error.response)
          })*/
        },
        getVideo({commit}){
          axios.get('http://localhost:3000/events')
          .then(response => {

                commit('ADD_VIDEO',response.data)

          })
          .catch(error =>{
            console.log('Un message d erreur : ',error.response)
          })
        },
        getVideos({commit}){

          axios.get('http://localhost:3000/events')
          .then(response => {

                commit('ADD_VIDEOS',response.data)

          })
          .catch(error =>{
            console.log('Error getVideos: ',error.response)
          })
        }
  },
  getters:{
        getPath : state=>{
                return state.test
        }
  }

  //end export
})