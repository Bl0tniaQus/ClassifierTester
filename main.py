from flask import Flask, render_template, session, request, redirect, url_for
from flask_session import Session
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./tmp/"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
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
	return render_template("index.html")
@app.route("/dane", methods=["POST","GET"])
def dane():
	if request.method=="POST":
		p = request.files["dane"]
		nazwa_pliku = secure_filename(p.filename)
		p.save(app.config['UPLOAD_FOLDER']+nazwa_pliku)
		dane = np.loadtxt(app.config['UPLOAD_FOLDER'] + nazwa_pliku, delimiter=request.form["separator"], dtype='str')
		os.remove(app.config['UPLOAD_FOLDER'] + nazwa_pliku)
		session.clear()
		session['dane_raw'] = dane
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
			session['preview']=preview
			session['n_wierszy']=len(dane)
	if 'preview' in session:
		return render_template("dane.html",preview=session['preview'], n_wierszy=session['n_wierszy'])
	return render_template("dane.html")
@app.route("/klasyfikator", methods=["POST","GET"])
def klasyfikator():
	if request.method=="POST":
		dane=session['dane_raw']
		if not request.form.get('pwiersz'):
			dane = np.delete(dane, (0), axis=0)
		session['target_col'] = request.form["target"]
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
		session["dane"] = dane
		session["target"] = target
	return render_template("klasyfikator.html")
@app.route("/wynik", methods=["POST","GET"])
def wynik():
	if request.method=="POST":
		alg = request.form['algs']
		
		if alg=="Regresja logistyczna":
			tol = float(request.form['tol'])
			solver = request.form['method']
			max_iter = int(request.form['epoki'])
			if max_iter<=0:
				max_iter=100
			model = LogisticRegression(max_iter = max_iter, tol = tol, solver=solver).fit(session["dane"], session["target"])
			params='L. epok: '+str(max_iter)+'<br/>Tolerancja: '+str(tol)+'<br/>Solver: '+solver
		elif alg=="KNN":
			if request.form['n']:
				n = int(request.form['n'])
				if n<=0:
					n=3
			else:
				n = 3
			model = KNeighborsClassifier(n_neighbors=n).fit(session["dane"], session["target"])
			params = n
		elif alg=="SVM":
			kernel = request.form['kernel']
			model = SVC(kernel=kernel).fit(session["dane"], session["target"])
			params = 'Jądro: '+str(kernel)
			
		result = model.predict(session["dane"])
		t = sorted(list(set(session["target"])))
		cm = confusion_matrix(result,session["target"],labels=t)
		confm = '<table><tr><td>T\\P</td>'
		for i in t:
			confm=confm+'<td>'+i+'</td>'
		confm +='</tr>'
		for j in range(len(cm)):
			confm=confm+'<tr><td>'+t[j]+'</td>'
			for k in cm[j]:
				confm=confm+'<td>'+str(k)+'</td>'
			confm+='</tr>'
		confm+='</table>'
		session['confm']=confm
		session['params']=params
		session['alg']=alg
		session['acc']=accuracy_score(result,session["target"])
		session['rec']=recall_score(result,session["target"],average='weighted')
		session['prec']=precision_score(result,session["target"],average='weighted')
	if 'confm' in session:
		return render_template("wynik.html", acc=session['acc'],rec=session['rec'],prec=session['prec'],confm=session['confm'],alg=session['alg'],params=session['params'])
	return render_template("wynik.html")
@app.route("/wyczysc")
def wyczysc():
	session.clear()
	return redirect("/")
if __name__ =="__main__":
	app.run(debug=True)



