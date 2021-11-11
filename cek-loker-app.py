import pandas as pd
import streamlit as st
import numpy as np
import time
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
#from googletrans import Translator

def translate_input(title, company_profile, description, requirements, benefits):
	#translator =  Translator()
	#english_title = translator.translate(title, dest = 'en')
	#english_company_profile = translator.translate(company_profile, dest = 'en')
	#english_description = translator.translate(description, dest = 'en')
	#english_requirements = translator.translate(requirements, dest = 'en')
	#english_benefits = translator.translate(benefits, dest = 'en')

	english_title_value = title
	english_company_profile_value = company_profile
	english_description_value = description
	english_requirements_value = requirements
	english_benefits_value = benefits

	return english_title_value, english_company_profile_value, english_description_value, english_requirements_value, english_benefits_value


def read_func_csv():
	temp_df = pd.read_csv('resources/function.csv')
	func_arry = temp_df['function'].unique()
	return func_arry

def read_industry_csv():
	temp_df = pd.read_csv('resources/industry.csv')
	ind_arr = temp_df['industry'].unique()
	return ind_arr

def load_stuff(label_col):
	temp_arr_label = []
	with open ('resources/LOKI_model_RFC.pkl', 'rb') as f:
		temp_model = pickle.load(f)
	with open ('resources/Tfidf_vect.pkl', 'rb') as f:
		temp_tfidf = pickle.load(f)
	for x in label_col:
		with open ('resources/label_' + x + '.pkl', 'rb') as f:
			temp_arr_label.append(pickle.load(f))
	return temp_model, temp_tfidf, temp_arr_label

def teks_preprocessing(df_teks):
	df_res = df_teks
	#LOWER TEKS
	for x in list(df_teks.columns):
		df_res[x] = [entry.lower() for entry in df_res[x]]
	#TOKENIZE
	for x in list(df_teks.columns):
		temp_arr = []
		for entry in df_res[x]:
			words = word_tokenize(entry)
			new_words= [word for word in words if word.isalnum()]
			temp_arr.append(new_words)
		df_res[x] = temp_arr
	#REMOVE STOP WORDS AND LEMMA
	# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
	tag_map = defaultdict(lambda : wn.NOUN)
	tag_map['J'] = wn.ADJ
	tag_map['V'] = wn.VERB
	tag_map['R'] = wn.ADV
	for x in list(df_teks.columns):
		for index,entry in enumerate(df_res[x]):
	        # Declaring Empty List to store the words that follow the rules for this step
			Final_words = []
	        # Initializing WordNetLemmatizer()
			word_Lemmatized = WordNetLemmatizer()
	        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
			for word, tag in pos_tag(entry):
	            # Below condition is to check for Stop words
				if word not in stopwords.words('english'):
					word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
					Final_words.append(word_Final)
			df_res.loc[index, x] = str(Final_words)
	return df_res

def teks_vectorization(df_teks,vectorizer):
	df_res = df_teks
	for x in list(df_teks.columns):
		X_teks = vectorizer.transform(df_res[x])
	X_tfidf = pd.DataFrame(X_teks.toarray())
	df_res = pd.DataFrame()
	df_res["sum"] = X_tfidf.sum(axis = 1)
	return df_res

def labeling(df_num, cols, arr_label):
	df_res = df_num
	i = 0
	for x in cols:
		le = arr_label[i]
		print(le.classes_)
		df_res[x] = le.transform(df_res[x])
		i = i + 1
	return df_res


st.title("Cek Lowongan Pekerjaanmu")
st.write('')
st.write("Periksa keaslian lowongan pekerjaan yang Anda temukan di internet sebelum melamar!")
st.write("Kami mengembangkan sebuah model Machine Learning bernama LOKI yang dapat memprediksi lowongan pekerjaan asli atau palsu.")
st.subheader("Lengkapi isian dibawah untuk memeriksa Lowongan Pekerjaanmu!")
st.write('Isian dapat diisi menggunakan Bahasa Indonesia atau Bahasa Inggris.')
st.write('Lakukan pencarian Google terhadap nama perusahaan di lowongan pekerjaan.')
st.write('Isian seperti: Profil perusahaan, Logo perusahaan, dan Industri perusahaan biasanya dapat diperoleh dengan pencarian Google.')
st.write('Apabila perusahaan tidak ditemukan, lanjut mengisi isian hanya dengan menggunakan informasi yang dicantumkan di lowongan pekerjaan.')
# Add a selectbox to the sidebar:
st.sidebar.header("Terima Kasih Sudah Menggunakan LOKI!")
st.sidebar.write("Website ini adalah sebuah prototipe yang mengimplementasikan model Machine Learning kami yaitu LOKI.")
st.sidebar.write("Model Machine Learning dibuat menggunakan dataset: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction")
st.sidebar.write("Semoga hasil dari LOKI dapat menambah pertimbangan Anda sebelum melamar pekerjaan.")
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.subheader("Contact us at: ahimsa.imananda@gmail.com")

title_job = st.text_input("Nama Lowongan Pekerjaan:",value = 'None')
st.text('Contoh: Marketing, Sales, Operator Produksi, Data Scientist, dll.')
st.text("")
company_profile = st.radio("Apakah perusahaan di lowongan pekerjaan memiliki profil perusahaan?", ('Ya', 'Tidak'), index = 1)
if company_profile == 'Ya':
	company_profile_value = st.text_area("Profil perusahaan:")
else:
	company_profile_value = "None"

st.text("")

description = st.radio("Apakah lowongan pekerjaan memiliki deskripsi pekerjaan?", ('Ya', 'Tidak'), index = 1)
if description == 'Ya':
	description_value = st.text_area("Deskripsi Pekerjaan:")
else:
	description_value = "None"
st.text("")

requirements = st.radio("Apakah lowongan pekerjaan memiliki syarat pekerjaan?", ('Ya', 'Tidak'), index = 1)
if requirements == 'Ya':
	requirements_value = st.text_area("Syarat pekerjaan:")
else:
	requirements_value = "None"

st.text("")
benefits = st.radio("Apakah lowongan pekerjaan memiliki keuntungan pekerjaan?", ("Ya", "Tidak"), index = 1)
st.text("Contoh: Asuransi, Bonus, Pengalaman, Skill, dll.")
if benefits == 'Ya':
	benefits_value = st.text_area("Keuntungan pekerjaan:")
else:
	benefits_value = "None"

st.text("")

company_logo = st.radio("Apakah perusahaan di lowongan pekerjaan memiliki logo perusahaan?", ('Ya', 'Tidak'), index = 1)
if company_logo=='Ya':
	company_logo_value = 1
else:
	company_logo_value = 0

st.text('')
has_questions = st.radio("Apakah terdapat pertanyaan-pertanyaan di lowongan pekerjaan?", ('Ya', 'Tidak'), index = 1)
if has_questions=='Ya':
	has_questions_value = 1
else:
	has_questions_value = 0

st.text('')

tipe_pekerjaan = ['Tidak Ditampilkan','Kontrak', 'Full-time', 'Part-time', 'Sementara', 'Lainnya']
st.write('*)Pilih opsi "Tidak Ditampilkan" jika kategori isian dibawah tidak dicantumkan di lowongan pekerjaan.')
employment_type = st.selectbox("Tipe pekerjaan:", tipe_pekerjaan, index = 0)
if employment_type == 'Tidak Ditampilkan':
	employment_type = 'Unknown'
if employment_type == 'Kontrak':
	employment_type = 'Contract'
if employment_type == 'Sementara':
	employment_type = 'Temporary'
if employment_type == 'Lainnya':
	employment_type = 'Other'
st.text('')

pengalaman = ['Tidak Ditampilkan', 'Associate', 'Director', 'Entry level', 'Executive', 'Internship', 'Mid-Senior level']
required_experience = st.selectbox("Pengalaman yang dibutuhkan:", pengalaman, index = 0)
if required_experience == 'Tidak Ditampilkan':
	required_experience = 'Unknown'

st.text('')

education = ['Tidak Ditampilkan', 'Tidak Ditentukan/Bebas','Bersertifikat', 'Professional', 'SMA/SMK/Sederajat', 'SMK', 'Pernah Kuliah', 'D1-D4', 'S1', 'S2', 'S3']
education_values = st.selectbox("Pendidikan yang dibutuhkan:", education, index = 0)
if education_values == 'Tidak Ditampilkan':
	education_values = 'Unknown'
elif education_values == 'Tidak Ditentukan/Bebas':
	education_values = 'Unspecified'
elif education_values == 'SMA/SMK/Sederajat':
	education_values = 'High School or equivalent'
elif education_values == 'SMK':
	education_values = 'Vocational'
elif education_values == 'Pernah Kuliah':
	education_values = 'Some College Coursework Completed'
elif education_values == 'D1-D4':
	education_values = 'Associate Degree'
elif education_values == 'S1':
	education_values = 'Bachelor\'s Degree'
elif education_values == 'S2':
	education_values = 'Master\'s Degree'
elif education_values == 'S3':
	education_values = 'Doctorate'
elif education_values == 'Bersertifikat':
	education_values = 'Certification'

st.text('')

industry = read_industry_csv()
industry_values = st.selectbox("Industri perusahaan:", industry, index = 0)
if industry_values == 'Tidak Diketahui / Lainnya':
	industry_values = 'Unknown'


st.text('')

function = read_func_csv()
function_values = st.selectbox("Fungsi pekerjaan:", function, index = 0)
if function_values == 'Tidak Diketahui / Lainnya':
	function_values = 'Unknown'


label_col = ['employment_type', 'required_experience', 'required_education', 'function', 'industry']
model, tfidf, arr_label = load_stuff(label_col)
st.write('')

title_job, company_profile_value, description_value, requirements_value, benefits_value = translate_input(title_job, company_profile_value, description_value, requirements_value, benefits_value)

data_teks = {'title': [title_job], 'company_profile': [company_profile_value], 'description': [description_value], 'requirements': [requirements_value], 'benefits': [benefits_value]}
data_num = {'has_company_logo' : [company_logo_value], 'has_questions': [has_questions_value], 'employment_type': [employment_type],'required_experience': [required_experience],'required_education': [education_values], 'industry': [industry_values], 'function':[function_values]}


df_teks = pd.DataFrame(data = data_teks)
df_num = pd.DataFrame(data = data_num)
df_all = pd.concat([df_teks, df_num], axis = 1)


if st.button('Submit'):
	with st.spinner('LOKI Sedang memeriksa masukkan pengguna. Mohon tunggu sebentar....'):
		my_bar = st.progress(0)
		for percent_complete in range(100):
			time.sleep(0.02)
			my_bar.progress(percent_complete + 1)	
	st.header('Hasil pemeriksaan lowongan pekerjaan menggunakan LOKI:')
	df_teks = teks_preprocessing(df_teks)
	df_sum = teks_vectorization(df_teks, tfidf)
	df_num_preprocessed = labeling(df_num, label_col, arr_label)
	df_final = pd.concat([df_sum, df_num_preprocessed], axis = 1)
	pred = model.predict(df_final)
	if pred == 1:
		st.error('Lowongan pekerjaan PALSU')
		my_bar.progress(0)
	elif pred == 0:
		st.success('Lowongan pekerjaan ASLI')
		my_bar.progress(0)

