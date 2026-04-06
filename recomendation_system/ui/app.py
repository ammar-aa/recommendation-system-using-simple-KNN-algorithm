import streamlit as st
import pandas as pd
import pickle


with open("book_recommender.pkl", "rb") as f:
    pipeline = pickle.load(f)

df = pd.read_csv("books.csv")
df.drop('Awards', axis=1, inplace=True)


allowed_features = ['Format', 'Theme (Color Style)', 'Genre','Language','Age Rating']
title_col = 'Title'
meta_data = ['Writer', 'Artist', 'Studio/Publisher','Release Year','Rating (out of 10)', 'Status','Country of Origin','Page Count','Volume Count']


st.title("Our database")
st.dataframe(df)

st.title("Book Recommendation System")
genre = st.selectbox("Genre", df['Genre'].unique())
language = st.selectbox("Language", df['Language'].unique())
book_format = st.selectbox("Format", df['Format'].unique())
theme = st.selectbox("Theme (Color Style)", df['Theme (Color Style)'].unique())
age_rating = st.selectbox("Age Rating", df['Age Rating'].unique())


if st.button("Recommend"):
    user_input = pd.DataFrame([{
        'Genre': genre,
        'Language': language,
        'Format': book_format,
        'Theme (Color Style)': theme,
        'Age Rating': age_rating,
    }])
    strict_filtered = df[
        (df['Genre'] == genre) &
        (df['Language'] == language) &
        (df['Format'] == book_format) &
        (df['Theme (Color Style)'] == theme) &
        (df['Age Rating'] == age_rating) 
    ]
    if not strict_filtered.empty:
     
     recommendations = strict_filtered.copy()
     st.success("Exact matches found!")
    else:
      
        distances, indices = pipeline.named_steps['knn'].kneighbors(
            pipeline.named_steps['preprocessing'].transform(user_input)
        )
        recommendations = df.iloc[indices[0]].copy()
        recommendations['similarity_distance'] = distances[0]
        st.info("No exact matches, showing closest books instead.")

    cols_to_show = [title_col] + allowed_features + meta_data
    if 'similarity_distance' in recommendations.columns:
        cols_to_show += ['similarity_distance']

    st.subheader("Recommended Books")
    st.dataframe(recommendations[cols_to_show])
