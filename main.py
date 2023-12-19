from flask import Flask, render_template, session, request, redirect, url_for, send_file
from flask_session import Session
import numpy as np
import os
import platform
import webbrowser
from copy import copy as copyf
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename
from sklearn.model_selection import StratifiedKFold


platform_name = platform.system()
app = Flask(__name__)
if platform_name == "Windows":
	app.config['UPLOAD_FOLDER'] = ".\\tmp\\"
else:	
	app.config['UPLOAD_FOLDER'] = "./tmp/"
if not os.path.exists(app.config['UPLOAD_FOLDER']):
		os.mkdir(app.config['UPLOAD_FOLDER'])
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

cv_splitter = StratifiedKFold(n_splits = 10)
njobs = -3

Session(app)

class Result:
	filename=""
	dane_raw=[]
	preview=""
	n_wierszy=0
	target_col=0
	data_cols = []
	norm_interval = []
	first_row = ""
	empty_data = ""
	dane=[]
	target=[]
	confm=""
	params=""
	report=""
	alg=""
	class_distribution=""
	acc=0
	rec=0
	prec=0

def normalizuj(x,nmin,nmax,vmin,vmax):
	return (nmax - (nmin)) * (x - vmin) / ((vmax - vmin)+1*10**-100) + (nmin)
def usunBraki(dane,target):
	puste=[]
	for x in range(len(dane)):
		if '' in dane[x]:
			print(dane[x])
			puste.append(x)
	dane = np.delete(dane, puste.copyf(), axis=0)
	target = np.delete(target,puste,axis=0)
	return [dane,target]
def usunPusteKlasy(dane):
	puste=[]
	for x in range(len(dane)):
		if dane[x][len(dane[x])-1]=='' or dane[x][len(dane[x])-1]=='?' or dane[x][len(dane[x])-1]=='-':
			puste.append(x)
	dane = np.delete(dane, puste, axis=0)
	return dane
def etykietuj(dane):
	temp = []
	dane = dane.transpose()
	for row in dane:
		rowset = set(row)
		rowset.discard('')
		values = list(rowset)
		for y in values:
			if y=='':
				continue
			if (not y.replace(".","").replace(",","").isnumeric()):
				etykiety = range(len(values))
				for x in range(len(row)):
					if row[x]=='':
						continue
					row[x] = etykiety[values.index(row[x])]
				break
		temp.append(row)
	return np.array(temp).transpose()
def usrednijBraki(dane,target):
	targetvals = list(set(target))
	counts=[]
	sums = []
	for i in range(len(targetvals)):
		v = []
		g = []
		for j in range(len(dane[0])):
			v.append(0)
			g.append(0)
		sums.append(v)
		counts.append(g)
	for row in range(len(dane)):
		for col in range(len(dane[row])):
			if dane[row][col]!='':
				i = targetvals.index(target[row])
				counts[i][col]+=1;
				c = col
				sums[i][c]+=float(dane[row][col]);
				
	for row in range(len(dane)):
		for col in range(len(dane[row])):
			if dane[row][col]=='':
				i = targetvals.index(target[row])
				dane[row][col]=sums[i][col]/counts[i][col]
	return dane

@app.route("/")
def index():
	if 'slot' not in session:
		session["slot"]=1
	if 'result1' not in session and 'result2' not in session:
		session["result1"] = Result()
		session["result2"] = Result()
	return render_template("index.html")
@app.route("/data", methods=["POST","GET"])
def data():
	if request.method=="POST":
		if "slot" in request.form:
			session["slot"] = int(request.form['slot'])
			return redirect("/")
		p = request.files["dane"]
		nazwa_pliku = secure_filename(p.filename)
		p.save(app.config['UPLOAD_FOLDER']+nazwa_pliku)
		
		dane = np.loadtxt(app.config['UPLOAD_FOLDER'] + nazwa_pliku, delimiter=request.form["separator"], dtype='str')
		os.remove(app.config['UPLOAD_FOLDER'] + nazwa_pliku)
		
		preview = '<table>'
		for y in range(len(dane[:3])):
			preview+='<tr>'
			
			if y==0:
				for v in range(len(dane[y])):
					preview=preview+'<td>Col '+str(v)+'</td>'
				preview+='</tr>'
			for x in dane[y]:
				preview=preview+'<td>'+x+'</td>'
			preview+='</tr>'
		if session["slot"]==1:
			session['result1'] = Result()
			session['result1'].preview=preview
			session['result1'].n_wierszy=len(dane)
			session['result1'].filename=nazwa_pliku
			session['result1'].dane_raw = dane
		elif session["slot"]==2:
			session['result2'] = Result()
			session['result2'].preview=preview
			session['result2'].n_wierszy=len(dane)
			session['result2'].filename=nazwa_pliku
			session['result2'].dane_raw = dane
	if 'preview' in session:
		return render_template("data.html")
	return render_template("data.html")
@app.route("/classifier", methods=["POST","GET"])
def classifier():
	if request.method=="POST":
		if "slot" in request.form:
			session["slot"] = int(request.form['slot'])
			return redirect("/dane")
		if session['slot']==1:
			dane=session['result1'].dane_raw
		elif session['slot']==2:
			dane=session['result2'].dane_raw
		if not request.form.get('pwiersz'):
			firstrow = False
			dane = np.delete(dane, (0), axis=0)
		else:
			firstrow = True
		
		puste = request.form["puste"]
		dane = usunPusteKlasy(dane)
		nmin = float(request.form["nmin"])
		nmax = float(request.form["nmax"])
		norm_interval = [nmin,nmax]
		target_col = int(request.form["target"])
		if target_col < 0 or target_col>=dane.shape[1]:
			target_col=dane.shape[1]-1
		target = dane[:,target_col];
		cols = request.form["columns"]
		data_cols = []
		for col in cols.strip().split(","):
			char = col.strip()
			if char.isnumeric():
				if int(char)>=0 and int(char) < dane.shape[1] and int(char)!=target_col:
					data_cols.append(int(char))
		data_cols = list(set(data_cols))
		if len(data_cols)==0:
			for x in range(dane.shape[1]):
				if x!=target_col:
					data_cols.append(x)
		
			
		#dane = np.delete(dane, int(request.form["target"]), 1)
		dane = dane[:,data_cols]
		dane = etykietuj(dane)
		if puste=="delete":
			res = usunBraki(dane,target)
			dane = res[0]
			target = res[1]
		elif puste=="avg":
			dane = usrednijBraki(dane,target)
		dane = dane.astype(float)
		danet = dane.transpose()
		temp = danet
		for row in range(temp.shape[0]):
			vec = []
			minv = min(temp[row])
			maxv = max(temp[row])
			for x in temp[row]:
				vec.append(normalizuj(x,nmin,nmax,minv,maxv))
			danet[row] = vec
		dane = danet.transpose()
		class_distribution =""
		for x in (sorted(list(set(target)))):
			class_distribution += str(x) + ": "+str(list(target).count(x))+"<br/>"
		
		if session["slot"]==1:
			session['result1'].target_col=target_col
			session['result1'].dane=dane
			session['result1'].target=target
			session['result1'].data_cols=data_cols
			session['result1'].confm=""
			session['result1'].params=""
			session['result1'].alg=""
			session['result1'].acc=0
			session['result1'].rec=0
			session['result1'].prec=0
			session['result1'].class_distribution=class_distribution
			session['result1'].norm_interval = norm_interval
			session['result1'].data_cols = data_cols
			session['result1'].first_row = firstrow
			session['result1'].empty_data = puste
		elif session["slot"]==2:
			session['result2'].target_col=target_col
			session['result2'].dane=dane
			session['result2'].target=target
			session['result2'].data_cols=data_cols
			session['result2'].confm=""
			session['result2'].params=""
			session['result2'].alg=""
			session['result2'].acc=0
			session['result2'].rec=0
			session['result2'].prec=0
			session['result2'].class_distribution=class_distribution
			session['result2'].norm_interval = norm_interval
			session['result2'].data_cols = data_cols
			session['result2'].first_row = firstrow
			session['result2'].empty_data = puste
	return render_template("classifier.html")
@app.route("/result", methods=["POST","GET"])
def result():
	if request.method=="POST":
		if "slot" in request.form:
			session["slot"] = int(request.form['slot'])
			return redirect("/classifier")
		if "slotw" in request.form:
			session["slot"] = int(request.form['slotw'])
			return redirect("/result")
		if "report" in request.form:
			
			if platform_name=="Windows":
				if not os.path.exists('.\\tmp'):
					os.mkdir('.\\tmp')
				if os.path.exists('.\\tmp\\report.txt'):
					os.remove('.\\tmp\\report.txt')
				if not os.path.exists('.\\tmp\\report.txt'):
					with open('.\\tmp\\report.txt', "w") as f:
						if session['slot']==1:
							f.write(session['result1'].report)
						elif session['slot']==2:
							f.write(session['result2'].report)
			else:
				if not os.path.exists('./tmp'):
					os.mkdir('./tmp')
				if os.path.exists('./tmp/report.txt'):
					os.remove('./tmp/report.txt')
				if not os.path.exists('./tmp/report.txt'):
					with open('./tmp/report.txt', "w") as f:
						if session['slot']==1:
							f.write(session['result1'].report)
						elif session['slot']==2:
							f.write(session['result2'].report)
			return send_file('./tmp/report.txt', as_attachment=True)
		alg = request.form['algs']
		if session['slot']==1:
			dane = session['result1'].dane
			target = session['result1'].target
		elif session['slot']==2:
			dane = session['result2'].dane
			target = session['result2'].target
		if alg=="LR":
			if 'auto' in request.form:
				param_grid = {'max_iter' : [5, 10, 25, 50, 100, 250, 500, 1000], 'tol': [0.1, 0.01, 0.001], 'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
				gridsearch = GridSearchCV(estimator=LogisticRegression(),param_grid=param_grid, cv=cv_splitter,n_jobs=njobs).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				max_iter = best_params["max_iter"]
				tol = best_params["tol"]
				solver = best_params["solver"]
			else:
				tol = float(request.form['tol'])
				solver = request.form['method']
				max_iter = int(request.form['epoki'])
				if max_iter<=0:
					max_iter=100
			model = LogisticRegression(max_iter = max_iter, tol = tol, solver=solver).fit(dane,target)
			params='Epochs: '+str(max_iter)+'<br/>Tolerancy: '+str(tol)+'<br/>Solver: '+solver
		elif alg=="KNN":
			if 'auto' in request.form:
				param_grid = {'n_neighbors' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,30,50,100]}
				gridsearch = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid, cv=cv_splitter,n_jobs=njobs).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				n = best_params["n_neighbors"]
			else:
				if request.form['n']:
					n = int(request.form['n'])
					if n<=0:
						n=3
				else:
					n = 3
				model = KNeighborsClassifier(n_neighbors=n).fit(dane,target)
			params = "n: "+str(n)
		elif alg=="SVM":
			if 'auto' in request.form:
				param_grid = {'max_iter' : [5, 10, 25, 50, 100, 250, 500, 1000], 'tol': [0.1, 0.01, 0.001], 'kernel' : ['linear', 'poly', 'sigmoid', 'rbf'], 'degree' : [2,3,4,5,6,7,8,9,10], 'C' : [0.5,1,3,5,10,20,50,100]}
				gridsearch = GridSearchCV(estimator=SVC(),param_grid=param_grid,n_jobs=njobs, cv=cv_splitter).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				max_iter = best_params["max_iter"]
				tol = best_params["tol"]
				kernel = best_params["kernel"]
				c = best_params["C"]
				degree = best_params["degree"]
			else:
				kernel = request.form['kernel']
				c = float(request.form['c'])
				max_iter = int(request.form['maxiter'])
				degree = int(request.form['degree'])
				if max_iter<-1:
					max_iter=-1
				if c<0:
					c=1
				if degree<1:
					degree=3
				tol = float(request.form['tol'])
				model = SVC(kernel=kernel,C=c,tol=tol,max_iter=max_iter,degree=degree).fit(dane,target)
			if (kernel=="poly"):
				params = 'Tolerancy: '+str(tol)+'<br/>Max iteration number: '+str(max_iter)+'<br/>C: '+str(c)+'<br/>Kernel: '+str(kernel)+'<br/>Polynominal degree: '+str(degree)
			else:
				params = 'Tolerancy: '+str(tol)+'<br/>Max iteration number: '+str(max_iter)+'<br/>C: '+str(c)+'<br/>Kernel: '+str(kernel);
		elif alg=="Dummy":
			if 'auto' in request.form:
				param_grid = {'strategy' : ['most_frequent', 'uniform', 'prior', 'stratified']}
				gridsearch = GridSearchCV(estimator=DummyClassifier(), param_grid=param_grid,n_jobs=njobs, cv=cv_splitter).fit(dane,target)
				model = gridsearch.best_estimator_
				strategy = gridsearch.best_params_["strategy"]
			else:
				strategy = request.form['strategy']
				model = DummyClassifier(strategy=strategy).fit(dane,target)
			params= "Strategy: "+strategy
		elif alg=="NB":
				model = GaussianNB().fit(dane,target)
				params = ""
		elif alg=="GBC":
			if 'auto' in request.form:
				param_grid = {'max_depth' : [1,3,5,10], 'tol' : [0.1, 0.01, 0.001], 'n_estimators' : [5,10,20,50,100,200,500,1000], 'learning_rate' : [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1], 'criterion' : ['friedman_mse', 'squared_error']}
				gridsearch = GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=param_grid,n_jobs=njobs, cv = cv_splitter).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				max_depth = best_params["max_depth"]
				tol = best_params["tol"]
				criterion = best_params["criterion"]
			else:
				learning_rate = float(request.form['learning_rate'])
				tol = float(request.form['tol'])
				max_depth = int(request.form['maxdepth'])
				n_estimators = int(request.form['nestimators'])
				criterion = request.form['criterion']
				model = GradientBoostingClassifier(max_depth = max_depth, learning_rate = learning_rate, tol = tol, criterion = criterion, n_estimators = n_estimators).fit(dane,target)
			params = "Tolerancy: "+str(tol)+"<br/>Learning rate: "+str(learning_rate)+"<br/>Criterion: "+criterion+"<br/> No. of trees: "+str(n_estimators)+"<br/>Max depth: "+str(max_depth)
		elif alg=="DT":
			if 'auto' in request.form:
				param_grid = {'criterion' : ["gini", "entropy"], 'splitter' : ["best","random"]}
				gridsearch = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=param_grid, cv=cv_splitter,n_jobs=njobs).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				criterion = best_params["criterion"]
				splitter = best_params["splitter"]
			else:
				criterion = request.form['criterion']
				splitter = request.form['splitter']
				model = DecisionTreeClassifier(criterion=criterion, splitter=splitter).fit(dane,target)
			params = "Criterion: "+criterion+"<br/> Splitter: "+splitter
		elif alg=="RF":
			if 'auto' in request.form:
				param_grid = {'criterion' : ["gini", "entropy"], 'max_depth' : [-1, 1, 2, 3, 5, 10, 20, 50, 100], 'n_estimators': [3, 10, 50, 100,200,500,1000, 2500, 5000]}
				gridsearch = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid, cv=cv_splitter,n_jobs=njobs).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				criterion = best_params["criterion"]
				max_depth = best_params["max_depth"]
				n_estimators = best_params["n_estimators"]
			else:
				criterion = request.form['criterion']
				n_estimators = int(request.form['n_estimators'])
				max_depth = int(request.form['maxdepth'])
				if max_depth<=0:
					max_depth = None
				model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators = n_estimators).fit(dane,target)
			params = "Criterion: "+criterion+"<br/> No. of trees: "+str(n_estimators)+"<br/>Max depth: "+str(max_depth)
		elif alg=="Ridge":
			if 'auto' in request.form:
				param_grid = {'max_iter' : [5, 10, 25, 50, 100, 250, 500, 1000], 'tol': [0.1, 0.01, 0.001], 'solver' : ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], "alpha" : [0.1,0.5,1,2,3,5,10]}
				gridsearch = GridSearchCV(estimator=RidgeClassifier(),param_grid=param_grid,n_jobs=njobs, cv=cv_splitter).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				max_iter = best_params["max_iter"]
				tol = best_params["tol"]
				solver = best_params["solver"]
				alpha = best_params["alpha"]
			else:
				solver = request.form['solver']
				alpha = float(request.form['alpha'])
				max_iter = int(request.form['maxiter'])
				if max_iter<-1:
					max_iter=-1
				if alpha<0:
					c=1
				tol = float(request.form['tol'])
				model = RidgeClassifier(solver=solver,alpha=alpha,tol=tol,max_iter=max_iter).fit(dane,target)
				params = 'Tolerancy: '+str(tol)+'<br/>Max iterations: '+str(max_iter)+'<br/>Alpha: '+str(alpha)+'<br/>Solver: '+str(solver);
		elif alg=="LDA":
			if 'auto' in request.form:
				param_grid = {'solver' : ['svd', 'eigen', 'lsqr'], 'tol': [0.1, 0.01, 0.001], 'shrinkage' : [None, "auto"]}
				gridsearch = GridSearchCV(estimator=LinearDiscriminantAnalysis(),param_grid=param_grid,n_jobs=njobs, cv=cv_splitter).fit(dane,target)
				model = gridsearch.best_estimator_
				best_params = gridsearch.best_params_
				shrinkage = best_params["shrinkage"]
				tol = best_params["tol"]
				solver = best_params["solver"]
			else:
				solver = request.form['solver']
				shrinkage = request.form['shrinkage']
				if shrinkage=="None":
					shrinkage = None
				tol = float(request.form['tol'])
				model = LinearDiscriminantAnalysis(solver=solver,shrinkage=shrinkage,tol=tol).fit(dane,target)
				if shrinkage == "auto":
					shrinkage = "Ledoit-Wolf"
			params = 'Tolerancy: '+str(tol)+'<br/>Solver: '+str(solver)+'<br/>Shrinkage: '+str(shrinkage);
		result = cross_val_predict(model, dane,target, cv=cv_splitter)
		#result = model.predict(dane)
		acc = accuracy_score(result, target)
		prec = precision_score(result, target, average="macro")
		rec = recall_score(result, target, average="macro")
		t = sorted(list(set(target)))
		cm = confusion_matrix(result,target,labels=t)
		confm = '<table><tr><td>P\\T</td>'
		confm_report = ' '
		for i in t:
			confm=confm+'<td>'+i+'</td>'
			confm_report=confm_report+' '+i
		confm +='</tr>'
		confm_report+='\n'
		for j in range(len(cm)):
			confm=confm+'<tr><td>'+t[j]+'</td>'
			confm_report=confm_report+t[j];
			for k in cm[j]:
				confm=confm+'<td>'+str(k)+'</td>'
				confm_report=confm_report+' '+str(k)
			confm+='</tr>'
			confm_report=confm_report+'\n'
		confm+='</table>'
		report=''
		
		
		
		if session["slot"]==1:
			session['result1'].confm=confm
			session['result1'].params=params
			session['result1'].alg=alg
			session['result1'].acc=round(acc,5)
			session['result1'].rec=round(rec,5)
			session['result1'].prec=round(prec,5)
			
			report+='Dataset: '+session['result1'].filename+'\n'
			report+='Data columns: '+str(session['result1'].data_cols)+'\n'
			report+='Target column: '+str(session['result1'].target_col)+'\n'
			report+='Empty: '+session['result1'].empty_data+'\n'
			report+='First row: '+str(session['result1'].first_row)+'\n'
			report+='Normalisation interval: '+str(session['result1'].norm_interval)+'\n'
			report+='Method: '+session['result1'].alg+'\n'
			report+='Params: \n'+str(session['result1'].params).replace("<br/>",'\n')+'\n'
			report+='\nAccuracy: '+str(session['result1'].acc)+'\n'
			report+='Precision: '+str(session['result1'].prec)+'\n'
			report+='Recall: '+str(session['result1'].rec)+'\n'
			report+='\nConfusion Matrix P\\T\n'+confm_report;
			session['result1'].report = report
			
			
		elif session["slot"]==2:
			session['result2'].confm=confm
			session['result2'].params=params
			session['result2'].alg=alg
			session['result2'].acc=round(acc,5)
			session['result2'].rec=round(rec,5)
			session['result2'].prec=round(prec,5)
			
			report+='Dataset: '+session['result2'].filename+'\n'
			report+='Data columns: '+str(session['result2'].data_cols)+'\n'
			report+='Target column: '+str(session['result2'].target_col)+'\n'
			report+='Empty: '+session['result2'].empty_data+'\n'
			report+='First row: '+str(session['result2'].first_row)+'\n'
			report+='Normalisation interval: '+str(session['result2'].norm_interval)+'\n'
			report+='Method: '+session['result2'].alg+'\n'
			report+='Params: \n'+str(session['result2'].params).replace("<br/>",'\n')+'\n'
			report+='\nAccuracy: '+str(session['result2'].acc)+'\n'
			report+='Precision: '+str(session['result2'].prec)+'\n'
			report+='Recall: '+str(session['result2'].rec)+'\n'
			report+='\nConfusion Matrix P\\T\n'+confm_report;
			session['result2'].report = report
	return render_template("result.html")
@app.route("/wyczysc")
def wyczysc():
	session.clear()
	return redirect("/")
@app.route("/help", methods=["POST", "GET"])
def help():
	if "slot" in request.form:
			session["slot"] = int(request.form['slot'])
			return redirect("/help")
	return render_template("help.html")
@app.route("/copy", methods=["POST","GET"])
def copy():
	if request.method=="POST":
		
		if "slot" in request.form:
			session["slot"] = int(request.form['slot'])
			return redirect("/copy")
		
		if request.form['kopia']=="12":
			session['result2'] = copyf(session["result1"])
		elif request.form['kopia']=="21":
			session['result1'] = copyf(session["result2"])
	return render_template("copy.html")
if __name__ =="__main__":
	webbrowser.open("127.0.0.1:5000")
	app.run(debug=True)
	
	



