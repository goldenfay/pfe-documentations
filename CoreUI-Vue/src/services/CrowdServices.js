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
	
	getSensors(){ 
		return apiClient.get('/crowds')
	},
	postSensorData(data){

		apiClient.post('/crowds',data).
		then((result)=>{
			console.warn(result)
		})
	},
	postEditData(id,data){
		apiClient.put('/crowds/'+id,data).
		then((result)=>{
			console.warn(result)
		})
	},
	deleteSensorRequest(id){
		apiClient.delete('/crowds/'+id).
		then((result)=>{
			console.warn(result)
		})
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

