var curalg = "LR"
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
		else if (algs=="LR")
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
		else if (algs=="DT")
		{
		document.querySelector("#params").innerHTML = 'Criterion: <select name="criterion"><option value="gini" selected>Gini impurity</option><option value="log_loss">Cross-entropy loss</option></select><br/>Splitter: <select name="splitter"><option value="best">Best</option><option value="random">Random best</option></select>'
		curalg = algs
		}
		else if (algs=="RF")
		{
		document.querySelector("#params").innerHTML = 'No. of trees: <input type="number" name="n_estimators" value="100" min="1" required><br/>Max depth: (0 or negative for none)<input type="number" name="maxdepth" value="-1" required><br/>Criterion: <select name="criterion"><option value="gini" selected>Gini impurity</option><option value="log_loss">Cross-entropy loss</option></select><br/>'
		curalg = algs
		}
		else if (algs=="Ridge")
		{
		document.querySelector("#params").innerHTML = '<table><tr><td>Alpha: </td><td><input type="number" name="alpha" value="1" min="0" step="any" required></td></tr><tr><td>Tolerancy: </td><td><input type="number" name="tol" value=0.0001 step="any" required></td></tr><tr><td>Max iter(-1 - no limit): </td><td><input type="number" name="maxiter" value="100" required></td></tr><tr><td>Solver: </td><td><select name="solver"><option value="svd" selected>svd</option><option value="cholesky">cholesky</option><option value="lsqr">lsqr</option><option value="sparse_cg">sparse_cg</option><option value="sag">sag</option><option value="saga">saga</option></select></td></tr></table>'
		curalg = algs
		}
		else if (algs=="LDA")
		{
		document.querySelector("#params").innerHTML = '<table><tr><td>Solver: </td><td><select name="solver"><option value="svd" selected>svd</option><option value="lsqr">lsqr</option><option value="eigen">eigen</option></select></td><tr><td>Shrinkage: </td><td><select name="shrinkage"><option value="None" selected>None</option><option value="auto">Ledoit-Wolf</option></select></td></tr></tr><tr><td>Tolerancy: </td><td><input type="number" name="tol" value=0.0001 step="any" required></td></tr></table>'
		curalg = algs
		}
		}
},10)
	
	
}
