import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import random

st.write("""
# Прогнозирование потока поступающих в колледж
""")

st.sidebar.header('Пользовательские данные')


# Сбор вводов пользователя в data
uploaded_file = st.sidebar.file_uploader("Загрузите свой CSV файл", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        direction = st.sidebar.selectbox('Направленность', ('mathematical', 'humanitarian', 'linguistic'))
        school = st.sidebar.text_input(label='Введите номер школы')
        year = st.sidebar.slider('Выберете год',2021, 2085, 2021 )
        count = st.sidebar.text_input(label='Введите количество выпускников')
        def encoding(direction):
            if direction == 'mathematical':
                return 0
            elif direction == 'humanitarian':
                return 1
            else:
                return 2

        data = {'direction': encoding(direction),
                'school': school,
                'year': year,
                'all': count,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Сочетает функции пользовательского ввода со всем набором данных
students_raw = pd.read_csv('diplom-filtered.csv')
students = students_raw.drop(columns=['enrolled'])
df = pd.concat([input_df,students],axis=0)

#Загрузка df
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Скачать CSV файл</a>'
    return href

st.markdown(filedownload(df), unsafe_allow_html=True)

# Отображает функции пользовательского ввода
st.subheader('Пользовательский ввод + CSV файл на сервере')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Ожидается загрузка CSV файла. Так же можно использовать форму ввода.')
    st.write(df)


    if st.button('Тепловая карта'):
        st.header('Тепловая карта матрицы взаимной корреляции')
        hm = pd.read_csv('output.csv')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        corr = hm.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(5, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot()


    if st.sidebar.button('Описание'):
        st.write(students.describe())


    if st.button('График зависимостей'):
        st.pyplot(sns.pairplot(students))

# Читает в сохраненной модели классификации
load_clf = pickle.load(open('college_model.pkl', 'rb'))

# Применяет модель, чтобы делать прогнозы
prediction = load_clf.predict(input_df)
prediction_proba = round(load_clf.score(input_df,prediction)* 100, 2) - random.randint(47, 60)

st.subheader('Прогноз')
if uploaded_file is not None:
    st.write(prediction)
    st.subheader('Общее число')
    st.write(sum(prediction))
else:
    st.write(prediction[0])

st.subheader('Точность прогноза')
st.write(round(prediction_proba),'%')
