import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up Streamlit app



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set up Streamlit app

# Define function for first page
def new():
    st.title("New E-Commerce Platform")
    product_descriptions = pd.read_csv('product_descriptions.csv')
    product_descriptions_file = st.file_uploader("Upload CSV", type=["csv"])
    if product_descriptions_file is not None:
        product_descriptions = pd.read_csv('product_descriptions.csv')


    # Drop any rows with missing values
    product_descriptions = product_descriptions.dropna()

    # Choose a subset of the dataset to work with

    product_descriptions1 = product_descriptions.head(500)
    button_click=st.button("Display Data")
    if(button_click):
        st.write(product_descriptions1)

    # Create a TF-IDF vectorizer object and fit it to the product descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    X1 = vectorizer.fit_transform(product_descriptions1["product_description"])

    # Define the number of clusters to use and fit a K-Means clustering model to the TF-IDF vectors
    true_k = 10
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X1)

    # Define a function to print the top terms for a given cluster
    def print_cluster(i):
        st.write("Cluster %d:" % i)
        cluster_items = []
        for ind in order_centroids[i, :10]:
            cluster_items.append(terms[ind])
        cluster_items_df = pd.DataFrame(cluster_items, columns=["Top items"])
        st.write(cluster_items_df)

    # Define a function to show recommendations for a given product
    def show_recommendations(product):
        Y = vectorizer.transform([product])
        prediction = model.predict(Y)
        print_cluster(prediction[0])

    # Print the top terms for each cluster
    # st.write("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    # for i in range(true_k):
    #     print_cluster(i)

    # Add a text field to enter the product name
    product_name = st.text_input('Enter a product name')

    # Show recommendations for the entered product name
    if product_name:
        st.write("Top cluster items for product '{}'".format(product_name))
        show_recommendations(product_name)


# Define function for second page
def old():
    st.title("Old E-Commerce Platform")


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


# Define main function
def main():
    st.title('Team17 Product Recommendation System🎳🛠️')
    st.sidebar.title("Choose Platform🗃️")
    page = st.sidebar.selectbox("Select a page", ["Old Platform", "New Platform"])
    if page == "Old Platform":
        old()
    else:
        new()

# Run main function
if __name__ == '__main__':
    main()
