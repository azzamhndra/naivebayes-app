from flask import Flask, render_template, request, session, redirect, url_for, flash
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os
import joblib
from text_preprocessing2 import TextPreprocessor
import tempfile
from joblib import Parallel, delayed

app = Flask(__name__)
app.secret_key = 'naivebayes-azza'

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/beranda')
def beranda():
    return render_template("home.html")

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    session.clear()

    file = request.files['file']
    filename = file.filename

    if not filename.endswith('.csv'):
        flash("File yang diunggah harus berformat .csv.", "401")
        return redirect(url_for('klasifikasi'))

    dataset = pd.read_csv(file)

    required_columns = {'ulasan', 'sentimen'}
    if not required_columns.issubset(dataset.columns):
        flash("Dataset tidak sesuai format. Pastikan memiliki kolom 'ulasan' dan 'sentimen'.", "402")
        return redirect(url_for('klasifikasi'))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    dataset.to_csv(temp_file.name, index=False)

    return redirect(url_for('result', temp_file_path=temp_file.name))



def preprocess_text(text, preprocessor):
    return ' '.join(preprocessor.preprocess(text))

def preprocess_dataset(dataset):
    preprocessor = TextPreprocessor(normalization_file="static/dataset/kamus_normalisasi.csv")
    texts = dataset['ulasan'].tolist()
    clean_texts = Parallel(n_jobs=-1)(delayed(preprocess_text)(text, preprocessor) for text in texts)
    dataset['clean_text'] = clean_texts
    return dataset

def train_evaluate_fold(args):
    train_index, test_index, texts, y = args
    X_train_raw = [texts[i] for i in train_index]
    X_test_raw = [texts[i] for i in test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model = MultinomialNB(alpha=0.6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    all_labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metriks_evaluasi = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'y_test': y_test,
        'y_pred': y_pred,
        'model': model,
        'vectorizer': vectorizer
    }

    return metriks_evaluasi

def classify_and_evaluate(dataset):
    dataset = preprocess_dataset(dataset)
    text_bersih = dataset['clean_text'].tolist()
    y = dataset['sentimen']

    metrics = cross_validate_model(text_bersih, y)

    save_model(metrics['best_model'], metrics['best_vectorizer'])

    metrics['text_bersih'] = text_bersih
    return metrics

def cross_validate_model(texts, y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    fold_tps, fold_tns, fold_fps, fold_fns = [], [], [], []
    y_true_all = []
    y_pred_all = []

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    best_model, best_vectorizer = None, None
    best_accuracy, best_f1_score, best_precision = 0, 0, 0

    args = [(train_index, test_index, texts, y) for train_index, test_index in kf.split(texts, y)]
    
    with Parallel(n_jobs=-1) as parallel:
        results = parallel(delayed(train_evaluate_fold)(arg) for arg in args)

    for metriks_evaluasi in results:
        acc = metriks_evaluasi['accuracy']
        prec = metriks_evaluasi['precision']
        rec = metriks_evaluasi['recall']
        f1 = metriks_evaluasi['f1_score']
        tp = metriks_evaluasi['tp']
        tn = metriks_evaluasi['tn']
        fp = metriks_evaluasi['fp']
        fn = metriks_evaluasi['fn']
        y_test = metriks_evaluasi['y_test']
        y_pred = metriks_evaluasi['y_pred']
        model = metriks_evaluasi['model']
        vectorizer = metriks_evaluasi['vectorizer']

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        fold_tps.append(tp)
        fold_tns.append(tn)
        fold_fps.append(fp)
        fold_fns.append(fn)

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        if acc > best_accuracy or (acc == best_accuracy and f1 > best_f1_score) or (acc == best_accuracy and f1 == best_f1_score and prec > best_precision):
            best_accuracy = acc
            best_f1_score = f1
            best_precision = prec
            best_model = model
            best_vectorizer = vectorizer

    avg_metrics = {
        'avg_accuracy': round(sum(accuracies) / len(accuracies), 2),
        'avg_precision': round(sum(precisions) / len(precisions), 2),
        'avg_recall': round(sum(recalls) / len(recalls), 2),
        'avg_f1_score': round(sum(f1_scores) / len(f1_scores), 2)
    }

    return {
        **avg_metrics,
        'total_tp': total_tp,
        'total_tn': total_tn,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'fold_tps': fold_tps,
        'fold_tns': fold_tns,
        'fold_fps': fold_fps,
        'fold_fns': fold_fns,
        'total_rows': len(y),
        'total_positive_predictions': sum(1 for pred in y_pred_all if pred == 1),
        'total_negative_predictions': sum(1 for pred in y_pred_all if pred == 0),
        'fold_accuracies': [round(acc, 2) for acc in accuracies],
        'fold_precisions': [round(prec, 2) for prec in precisions],
        'fold_recalls': [round(rec, 2) for rec in recalls],
        'fold_f1_scores': [round(f1, 2) for f1 in f1_scores],
        'y_pred': y_pred_all,
        'best_model': best_model,
        'best_vectorizer': best_vectorizer
    }

def save_model(model, vectorizer):
    os.makedirs('static/model', exist_ok=True)
    joblib.dump(model, 'static/model/model_naivebayes.pkl')
    joblib.dump(vectorizer, 'static/model/tfidf_vectorizer.pkl')

NEGASI = {'tidak', 'bukan', 'belum', 'tak', 'jangan', 'ga', 'gak',
          'nggak', 'enggak', 'ndak', 'kurang', 'tiada'}

def negation_handling(text):
    words = text.lower().split()
    return any(word in NEGASI for word in words)

@app.route('/klasifikasi-manual', methods=['POST'])
def klasifikasi_manual():
    data = request.get_json()
    text_input = data.get('text', '').strip()

    model_path = 'static/model/model_naivebayes.pkl'
    vectorizer_path = 'static/model/tfidf_vectorizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return {'sentimen': 'Model atau Vectorizer tidak ditemukan'}, 500

    preprocessor = TextPreprocessor(normalization_file="static/dataset/kamus_normalisasi.csv")
    clean_text = preprocess_text(text_input, preprocessor)

    print(clean_text)

    tfidf_vectorizer = joblib.load(vectorizer_path)
    text_input_transformed = tfidf_vectorizer.transform([clean_text])

    model = joblib.load(model_path)
    prediction = model.predict(text_input_transformed)[0]
    confidence_score = model.predict_proba(text_input_transformed)[0][prediction]

    sentiment = 'Positif' if prediction == 1 else 'Negatif'

    return {'sentimen': sentiment, 'confidence_score': confidence_score}

def generate_wordcloud(text):
    if not text.strip():
        return None

    wordcloud = WordCloud(width=482, height=250, background_color='#161D36').generate(text)
    wordcloud_image_path = 'static/images/wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)
    return wordcloud_image_path

def generate_pie_chart(y_pred):
    positive = sum(1 for p in y_pred if p == 1)
    negative = len(y_pred) - positive

    if positive + negative == 0:
        positive, negative = 1, 1

    fig, ax = plt.subplots(figsize=(3.39, 2.5))
    fig.patch.set_facecolor('#161D36')
    ax.set_facecolor('#161D36')

    colors = ['#6E29D0', '#D02929']
    labels = ['Positif', 'Negatif']
    sizes = [positive, negative]

    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           startangle=90, textprops={'color': 'white'}, colors=colors)
    ax.axis('equal')

    legend_labels = [f'{label}: {count}' for label, count in zip(labels, sizes)]
    ax.legend(legend_labels, loc='upper center', fontsize=8, frameon=False, labelcolor=['white', 'white'], 
              facecolor='#161D36', bbox_to_anchor=(0.5, 0), ncol=2)

    plt.tight_layout(pad=0)

    pie_chart_image_path = 'static/images/pie_chart.png'
    os.makedirs(os.path.dirname(pie_chart_image_path), exist_ok=True)
    plt.savefig(pie_chart_image_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return pie_chart_image_path

def generate_bar_chart(fold_accuracies):
    folds = list(range(1, len(fold_accuracies) + 1))

    fig, ax = plt.subplots(figsize=(5.54, 2.5))
    fig.patch.set_facecolor('#161D36')
    ax.set_facecolor('#161D36')

    bar_width = 0.7
    bars = ax.bar(folds, fold_accuracies, color='#6E29D0', width=bar_width, zorder=3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.03, f'{height:.2f}', ha='center', va='bottom', color='#FFFFFF', fontsize=11, zorder=4)

    ax.set_xticks(folds)
    ax.set_xticklabels([f'{i}' for i in folds], color='#FFFFFF', rotation=0, fontsize=11)
    ax.set_ylabel('Accuracy', color='#FFFFFF', labelpad=10, fontsize=11)
    ax.set_xlabel('Fold Ke-', color='#FFFFFF', labelpad=10, fontsize=11)
    ax.tick_params(axis='y', colors='#FFFFFF', labelsize=11)
    ax.tick_params(axis='x', colors='#FFFFFF', labelsize=11)
    ax.spines['top'].set_color('#161D36')
    ax.spines['right'].set_color('#161D36')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', color='#555555', linestyle='--', linewidth=0.5, zorder=1)

    bar_chart_image_path = 'static/images/bar_chart.png'
    plt.tight_layout(pad=0)
    plt.savefig(bar_chart_image_path, format='png', bbox_inches='tight')
    plt.close(fig)
    return bar_chart_image_path

@app.route('/result')
def result():

    temp_file_path = request.args.get('temp_file_path')
    if not temp_file_path or not os.path.exists(temp_file_path):
        return "Dataset tidak ditemukan." #Check

    dataset = pd.read_csv(temp_file_path)
    metrics = classify_and_evaluate(dataset)

    pie_chart_url = generate_pie_chart(metrics['y_pred'])
    wordcloud_url = generate_wordcloud(' '.join(metrics['text_bersih']))
    bar_chart_url = generate_bar_chart(metrics['fold_accuracies'])

    return render_template('result.html', **metrics, pie_chart_url=pie_chart_url, wordcloud_url=wordcloud_url, bar_chart_url=bar_chart_url)

if __name__ == '__main__':
    app.run(port=5002)