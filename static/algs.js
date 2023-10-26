var curalg = "Regresja logistyczna"
window.onload = function()
{
	setInterval(function(){
		let algs = document.querySelector("#algs").value;
		if (curalg!=algs){
		if (algs=="KNN") 
		{
			document.querySelector("#params").innerHTML = 'n: <input type="number" name="n" value="3" required>';
			curalg = algs;
		}
		else if (algs=="Regresja logistyczna")
		{
			document.querySelector("#params").innerHTML = 
			'Epoki: <input type="number" name="epoki" value="100" required><br/>Tolerancja: <input type="number" name="tol" value=0.0001 step="any" required> <br/>Metoda: <select name="method"><option value="lbfgs" selected>lbfgs</option><option value="liblinear">liblinear</option><option value="newton-cg">newton-cg</option><option value="newton-cholesky">newton-cholesky</option><option value="sag">sag</option><option value="saga">saga</option></select>';
			curalg = algs;
		}
		else if (algs=="SVM")
		{
			document.querySelector("#params").innerHTML = 'Jądro: <select name="kernel"required><option value="linear"selected>Liniowe</option><option value="poly">Wielomianowe</option></select>'	
			curalg = algs
		}
		
		}
},10)
	
	
}
