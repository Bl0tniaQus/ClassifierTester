from flask import Flask, render_template, session, request, redirect, url_for, send_file
from flask_session import Session
import numpy as np
import os
from copy import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from werkzeug.utils import secure_filename

#raporty
#gridsearchCV (auto dopasowanie)
#algorytmy

#lr
#ridge
#lda
#rf
#nb
#gbc
#knn
#dt
#dummy
#svm

#todo 14.11
#lepsze raporty
#algorytmy


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./tmp/"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

class Wynik:
	filename=""
	dane_raw=[]
	preview=""
	n_wierszy=0
	target_col=0
	dane=[]
	target=[]
	confm=""
	params=""
	raport=""
	alg=""
	acc=0
	rec=0
	prec=0

def normalizuj(x,nmin,nmax,vmin,vmax):
	return (nmax - (nmin)) * (x - vmin) / (vmax - vmin) + (nmin)
def usunBraki(dane,target):
	puste=[]
	for x in range(len(dane)):
		if '' in dane[x]:
			puste.append(x)
	dane = np.delete(dane, puste.copy(), axis=0)
	target = np.delete(target,puste,axis=0)
	return [dane,target]
def usunPusteKlasy(dane):
	puste=[]
	for x in range(len(dane)):
		if dane[x][len(dane[x])-1]=='':
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
			if (not y.isnumeric()):
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
	if 'wynik1' not in session and 'wynik2' not in session:
		session["wynik1"] = Wynik()
		session["wynik2"] = Wynik()
	return render_template("index.html")
@app.route("/dane", methods=["POST","GET"])
def dane():
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
					preview=preview+'<td>Kol '+str(v)+'</td>'
				preview+='</tr>'
			for x in dane[y]:
				preview=preview+'<td>'+x+'</td>'
			preview+='</tr>'
		print(Wynik())
		if session["slot"]==1:
			session['wynik1'] = Wynik()
			session['wynik1'].preview=preview
			session['wynik1'].n_wierszy=len(dane)
			session['wynik1'].filename=nazwa_pliku
			session['wynik1'].dane_raw = dane
		elif session["slot"]==2:
			session['wynik2'] = Wynik()
			session['wynik2'].preview=preview
			session['wynik2'].n_wierszy=len(dane)
			session['wynik2'].filename=nazwa_pliku
			session['wynik2'].dane_raw = dane
	if 'preview' in session:
		return render_template("dane.html")
	return render_template("dane.html")
@app.route("/klasyfikator", methods=["POST","GET"])
def klasyfikator():
	if request.method=="POST":
		if "slot" in request.form:
			session["slot"] = int(request.form['slot'])
			return redirect("/dane")
		if session['slot']==1:
			dane=session['wynik1'].dane_raw
		elif session['slot']==2:
			dane=session['wynik2'].dane_raw
		if not request.form.get('pwiersz'):
			dane = np.delete(dane, (0), axis=0)
		
		puste = request.form["puste"]
		dane = usunPusteKlasy(dane)
		nmin = float(request.form["nmin"])
		nmax = float(request.form["nmax"])
		target = dane[:,int(request.form["target"])];
		dane = np.delete(dane, int(request.form["target"]), 1)
		dane = etykietuj(dane)
		if puste=="usun":
			res = usunBraki(dane,target)
			dane = res[0]
			target = res[1]
		elif puste=="usrednij":
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
		if session["slot"]==1:
			session['wynik1'].target_col=request.form["target"]
			session['wynik1'].dane=dane
			session['wynik1'].target=target
			session['wynik1'].confm=""
			session['wynik1'].params=""
			session['wynik1'].alg=""
			session['wynik1'].acc=0
			session['wynik1'].rec=0
			session['wynik1'].prec=0
		elif session["slot"]==2:
			session['wynik2'].target_col=request.form["target"]
			session['wynik2'].dane=dane
			session['wynik2'].target=target
			session['wynik2'].confm=""
			session['wynik2'].params=""
			session['wynik2'].alg=""
			session['wynik2'].acc=0
			session['wynik2'].rec=0
			session['wynik2'].prec=0
	return render_template("klasyfikator.html")
@app.route("/wynik", methods=["POST","GET"])
def wynik():
	if request.method=="POST":
		if "slot" in request.form:
			session["slot"] = int(request.form['slot'])
			return redirect("/klasyfikator")
		if "slotw" in request.form:
			session["slot"] = int(request.form['slotw'])
			return redirect("/wynik")
		if "raport" in request.form:
			if not os.path.exists('./tmp'):
				os.mkdir('./tmp')
			if os.path.exists('./tmp/raport.txt'):
				os.remove('./tmp/raport.txt')
			if not os.path.exists('./tmp/raport.txt'):
				with open('./tmp/raport.txt', "w") as f:
					if session['slot']==1:
						f.write(session['wynik1'].raport)
					elif session['slot']==2:
						f.write(session['wynik2'].raport)
			return send_file('./tmp/raport.txt', as_attachment=True)
		alg = request.form['algs']
		if session['slot']==1:
			dane = session['wynik1'].dane
			target = session['wynik1'].target
		elif session['slot']==2:
			dane = session['wynik2'].dane
			target = session['wynik2'].target
		if alg=="Regresja logistyczna":
			if 'auto' in request.form:
				param_grid = {'max_iter' : [5, 10, 25, 50, 100, 250, 500, 1000], 'tol': [0.1, 0.01, 0.001], 'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
				gridsearch = GridSearchCV(estimator=LogisticRegression(),param_grid=param_grid).fit(dane,target)
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
			params='L. epok: '+str(max_iter)+'<br/>Tolerancja: '+str(tol)+'<br/>Solver: '+solver
		elif alg=="KNN":
			if 'auto' in request.form:
				param_grid = {'n_neighbors' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,30,50,100]}
				gridsearch = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid).fit(dane,target)
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
			params = n
		elif alg=="SVM":
			if 'auto' in request.form:
				param_grid = {'max_iter' : [5, 10, 25, 50, 100, 250, 500, 1000], 'tol': [0.1, 0.01, 0.001], 'kernel' : ['linear', 'poly', 'sigmoid', 'rbf'], 'degree' : [2,3,4,5,6,7,8,9,10], 'C' : [0.5,1,3,5,10,20,50,100]}
				gridsearch = GridSearchCV(estimator=SVC(),param_grid=param_grid,n_jobs=-3).fit(dane,target)
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
			params = 'Tolerancja: '+str(tol)+'<br/>Max. n iteracji: '+str(max_iter)+'<br/>C: '+str(c)+'<br/>Jądro: '+str(kernel)+'<br/>Stopień (dla poly): '+str(degree)
		elif alg=="Dummy":
			if 'auto' in request.form:
				param_grid = {'strategy' : ['most_frequent', 'uniform', 'prior', 'stratified']}
				gridsearch = GridSearchCV(estimator=DummyClassifier(), param_grid=param_grid,n_jobs=-3).fit(dane,target)
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
				gridsearch = GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=param_grid,n_jobs=-3).fit(dane,target)
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
			params=""
		result = model.predict(dane)
		t = sorted(list(set(target)))
		cm = confusion_matrix(result,target,labels=t)
		confm = '<table><tr><td>T\\P</td>'
		confm_raport = ' '
		for i in t:
			confm=confm+'<td>'+i+'</td>'
			confm_raport=confm_raport+' '+i
		confm +='</tr>'
		confm_raport+='\n'
		for j in range(len(cm)):
			confm=confm+'<tr><td>'+t[j]+'</td>'
			confm_raport=confm_raport+t[j];
			for k in cm[j]:
				confm=confm+'<td>'+str(k)+'</td>'
				confm_raport=confm_raport+' '+str(k)
			confm+='</tr>'
			confm_raport=confm_raport+'\n'
		confm+='</table>'
		raport=''
		if session["slot"]==1:
			session['wynik1'].confm=confm
			session['wynik1'].params=params
			session['wynik1'].alg=alg
			session['wynik1'].acc=round(accuracy_score(result,target),5)
			session['wynik1'].rec=round(recall_score(result,target,average='weighted'),5)
			session['wynik1'].prec=round(precision_score(result,target,average='weighted'),5)
			
			raport+='Dataset: '+session['wynik1'].filename+'\n'
			raport+='Method: '+session['wynik1'].alg+'\n'
			raport+='Params: '+str(session['wynik1'].params)+'\n'
			raport+='Accuracy: '+str(session['wynik1'].acc)+'\n'
			raport+='Precision: '+str(session['wynik1'].prec)+'\n'
			raport+='Recall: '+str(session['wynik1'].rec)+'\n'
			raport+='Confusion Matrix T\\P\n'+confm_raport;
			session['wynik1'].raport = raport
			
			
		elif session["slot"]==2:
			session['wynik2'].confm=confm
			session['wynik2'].params=params
			session['wynik2'].alg=alg
			session['wynik2'].acc=round(accuracy_score(result,target),5)
			session['wynik2'].rec=round(recall_score(result,target,average='weighted'),5)
			session['wynik2'].prec=round(precision_score(result,target,average='weighted'),5)
			
			raport+='Dataset: '+session['wynik2'].filename+'\n'
			raport+='Method: '+session['wynik2'].alg+'\n'
			raport+='Params: '+str(session['wynik2'].params)+'\n'
			raport+='Accuracy: '+str(session['wynik2'].acc)+'\n'
			raport+='Precision: '+str(session['wynik2'].prec)+'\n'
			raport+='Recall: '+str(session['wynik2'].rec)+'\n'
			raport+='Confusion Matrix T\\P\n'+confm_raport;
			session['wynik2'].raport = raport
	print(session['wynik1'].dane)
	print("----------------------------")
	print(session['wynik2'].dane)
	return render_template("wynik.html")
@app.route("/wyczysc")
def wyczysc():
	session.clear()
	return redirect("/")
@app.route("/kopiuj", methods=["POST","GET"])
def kopiuj():
	if request.method=="POST":
		if request.form['kopia']=="12":
			session['wynik2'] = copy(session["wynik1"])
		elif request.form['kopia']=="21":
			session['wynik1'] = copy(session["wynik2"])
	return render_template("kopiuj.html")
if __name__ =="__main__":
	app.run(debug=True)
	



