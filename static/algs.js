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
			document.querySelector("#params").innerHTML = 'C: <input type="number" name="c" value="1" min="0" step="any" required> <br/>Tolerancja: <input type="number" name="tol" value=0.0001 step="any" required> <br/>Max. n. iteracji(-1 - bez limitu): <input type="number" name="maxiter" value="100" required><br/>Jądro: <select name="kernel"required><option value="linear"selected>Liniowe (linear)</option><option value="poly">Wielomianowe (poly)</option><option value="sigmoid">Sigmoidalne (sigmoid)</option><option value="rbf">RBF</option></select><br/>St. wielomianu(dla poly): <input type="number" name="degree" value="3" min="1" step="1" required>'	
			curalg = algs
		}
		else if (algs=="Dummy")
		{
			document.querySelector("#params").innerHTML = 'Strategy: <select name="strategy"><option value="most_frequent">Most frequent</option><option value="prior">Prior</option><option value="stratified">Stratified</option><option value="uniform">Uniform</option></select>'	
			curalg = algs
		}
		else if (algs=="NB")
		{
			document.querySelector("#params").innerHTML = ''	
			curalg = algs
		}
		else if (algs=="GBC")
		{
			document.querySelector("#params").innerHTML = 'Learning rate: <input type="number" name="learning_rate" value="0.1" min="0" max="1"step="any" required> <br/>Tolerancja: <input type="number" name="tol" value=0.0001 step="any" required> <br/>Max depth: <input type="number" name="maxdepth" value="1" required><br/>N. estimators: <input type="number" name="nestimators" value="100" required><br/>Criterion: <select name="criterion"required><option value="friedman_mse"selected>Friedman_mse</option><option value="squared_error">squared_error</option>'	
			curalg = algs
		}
		}
},10)
	
	
}
