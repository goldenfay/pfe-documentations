import axios from 'axios'

const apiClient = axios.create({
	
	baseURL: `http://localhost:3000`,
	withCredentials: true,
	headers:{
		Accept: 'application/json',
		'Content-Type': 'application/json',
	}
})
const apiDash = axios.create({
	
	baseURL: `http://localhost:8050`,
	headers:{
		Accept: 'application/json',
		'Content-Type': 'application/json',
		'Access-Control-Allow-Origin': '*',
		'Access-Control-Allow-Credentials': 'false'
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
	},
	registerSensor(complete_data,name,type){
		var the_type_s = ""
		if (type.localeCompare("Scène large échelle") == 0){
			the_type_s = "SANet"
		}
		else{
			the_type_s = "mobileSSD"
		}
		var data = {}
		data = {...complete_data,'model_name':the_type_s}
		return apiDash.post('/sensors/register',data)
	},
	UpdateRegistredSensor(complete_data,name,type){
		var the_type_s = ""
		if (type.localeCompare("Scène large échelle") == 0){
			the_type_s = "SANet"
		}
		else{
			the_type_s = "mobileSSD"
		}
		var data = {}
		data = {...complete_data,'model_name':the_type_s}
		return apiDash.post('/sensors/update',data)
	},
	DeleteRegistredSensor(sensor){
		return apiDash.post('/sensors/delete',sensor)	
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

