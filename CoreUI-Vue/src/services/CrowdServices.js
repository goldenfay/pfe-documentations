import axios from 'axios'

const apiCilent = axios.create({
	
	baseURL: `http://localhost:3000`,
	withCredentials: false,
	headers:{
		Accept: 'application/json',
		'Content-Type': 'application/json'
	}
})

export default{

	/*getCapImage(){
		return apiCilent.get('http://localhost:3000/events')
			.then(response =>{
				console.log(response.data)
			})
			.catch(error =>{
				console.log(error)
			})
	}*/

	//end export
}