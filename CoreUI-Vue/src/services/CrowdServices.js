import axios from 'axios'

const apiClient = axios.create({
	
	baseURL: `http://localhost:3000`,
	withCredentials: true,
	headers:{
		Accept: 'application/json',
		'Content-Type': 'application/json'
	}
})
export default{
	
	getSensorsPagination(perPage, page){
		return apiClient.get('/crowds?_limit='+perPage+"&_page="+page)
	},
	getSensors(){ 
		return apiClient.get('/crowds')
	},
	postSensorData(data){

		return apiClient.post('/crowds',data)
	},
	postEditData(id,data){
		return apiClient.put('/crowds/'+id,data)
	},
	deleteSensorRequest(id){
		return apiClient.delete('/crowds/'+id)
	}



	/*getCapImage(){
		return apiCilent.get('http://localhost:3000/events')
			.then(response =>{
				console.log(response.data)
			})
			.catch(error =>{
				console.log(error)
			})
	}*/
}

