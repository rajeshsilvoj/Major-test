



# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Set up Streamlit app
# st.title('Amazon Beauty Product Recommendation System')
#
# # Allow user to upload dataset
# uploaded_file = st.file_uploader('Upload a CSV file', type='csv')
# if uploaded_file is not None:
#     # Load data
#     nrow_value=st.slider('Select a value',min_value=1,max_value=10000,step=10)
#     ratings_data = pd.read_csv(uploaded_file,nrows=nrow_value)
#     st.write(ratings_data.head(100))
#
#     # Create a pivot table
#     pivot_table = ratings_data.pivot_table(index='UserId', columns='ProductId', values='Rating')
#
#     # Fill missing values with zeros
#     pivot_table = pivot_table.fillna(0)
#
#     # Calculate cosine similarity matrix
#     cosine_similarities = cosine_similarity(pivot_table)
#
#     # Define a function to get similar users
#     def get_similar_users(user_id, n):
#         similar_users = cosine_similarities[user_id-1]
#         similar_users_ids = np.argsort(-similar_users)[1:n+1]
#         return similar_users_ids
#
#     # Define a function to recommend products
#     def recommend_products(user_id, n):
#         similar_users = get_similar_users(user_id, n)
#         products = {}
#         for i in similar_users:
#             products.update(dict(pivot_table.iloc[i-1][pivot_table.iloc[i-1] > 0]))
#         sorted_products = sorted(products.items(), key=lambda x: x[1], reverse=True)[:n]
#         return [i[0] for i in sorted_products]
#
#     # Show example usage
#     user_id = st.number_input('Enter user ID:', min_value=1, max_value=nrow_value, value=1)
#     n_recommendations = st.slider('Number of recommendations:', min_value=1, max_value=10, value=5)
#     if st.button('Get recommendations'):
#         recommended_products = recommend_products(user_id, n_recommendations)
#         st.write('Top {} recommended products for user {}:'.format(n_recommendations, user_id))
#         for product_id in recommended_products:
#             st.write('- {}'.format(product_id))


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up Streamlit app
st.title('Team17 Product Recommendation System🎳🛠️')

# Allow user to upload dataset
uploaded_file = st.file_uploader('Upload a CSV file🗃️', type='csv')
if uploaded_file is not None:
    # Load data
    nrow_value=st.slider('Select a value',min_value=1,max_value=10000,step=10)
    ratings_data = pd.read_csv(uploaded_file,nrows=nrow_value)
    st.write(ratings_data.head(nrow_value))

    # Create a pivot table
    pivot_table = ratings_data.pivot_table(index='UserId', columns='ProductId', values='Rating')

    # Fill missing values with zeros
    pivot_table = pivot_table.fillna(0)

    # Calculate cosine similarity matrix
    cosine_similarities = cosine_similarity(pivot_table)

    # Define a function to get similar users
    def get_similar_users(user_id, n):
        similar_users = cosine_similarities[user_id-1]
        similar_users_ids = np.argsort(-similar_users)[1:n+1]
        return similar_users_ids

    # Define a function to recommend products
    def recommend_products(user_id, n):
        user_index = pivot_table.index.get_loc(user_id)
        similar_users = get_similar_users(user_index, n)
        products = {}
        for i in similar_users:
            products.update(dict(pivot_table.iloc[i]))
        sorted_products = sorted(products.items(), key=lambda x: x[1], reverse=True)[:n]
        return [i[0] for i in sorted_products]

    # Show example usage
    user_id = st.text_input('Enter User ID⌨️')
    st.write("User rated products👇")
    search_value=user_id
    if search_value:
        matching_rows = ratings_data.loc[ratings_data['UserId'] == search_value]
        st.write(matching_rows)
    n_recommendations = st.slider('Number of recommendations:', min_value=1, max_value=10, value=5)
    if st.button('Get recommendations'):
        if user_id in pivot_table.index:
            recommended_products = recommend_products(user_id, n_recommendations)
            st.write('Top {} recommended products for user {}:'.format(n_recommendations, user_id))
            for product_id in recommended_products:
                st.write('- {}'.format(product_id))
        else:
            st.write('User ID not found in dataset')
