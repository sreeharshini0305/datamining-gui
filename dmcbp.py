import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Streamlit Page Config
st.set_page_config(page_title="Data Mining App", layout="wide")

# Function to normalize data
def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Function to impute missing data (imputation)
def impute_data(df):
    imputer = SimpleImputer(strategy='mean')  # Can be 'mean', 'median', 'most_frequent', or 'constant'
    df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    return df

# Function to apply K-Means
def kmeans_clustering(df, n_clusters):
    # Only select numeric columns for clustering
    feature_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[feature_columns]  # Use only numeric features for KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)
    return df, model, feature_columns

# Function to apply Decision Tree with Entropy
# Function to apply Decision Tree with Entropy
def decision_tree(df, target_column, max_depth=None):
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target variable

    # Ensure categorical columns are encoded
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Decision Tree Classifier with max depth and entropy as criterion
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42, class_weight='balanced')  # Use 'entropy' instead of 'gini'
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Check the confusion matrix to see any potential problems
    st.write("Confusion Matrix:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    return clf

# Streamlit App
st.title("Data Mining GUI")

# Step 1: Upload Dataset
st.sidebar.header("Step 1: Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Original Dataset")
    st.dataframe(df)

    # Step 2: Preprocessing
    st.sidebar.header("Step 2: Preprocessing")

    # Normalize Data
    if st.sidebar.button("Normalize Data"):
        df = normalize_data(df)
        st.write("### Normalized Dataset")
        st.dataframe(df)

    # Impute Data
    if st.sidebar.button("Impute Missing Data"):
        df = impute_data(df)
        st.write("### Imputed Dataset")
        st.dataframe(df)

    # Handle Categorical Data (Encoding)
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Step 3: Select Algorithm
    st.sidebar.header("Step 3: Select Algorithm")
    algo_choice = st.sidebar.radio(
        "Choose an algorithm",
        ["K-Means Clustering", "Decision Tree"]
    )

    # Parameters and Execution for Selected Algorithm
    if algo_choice == "K-Means Clustering":
        st.sidebar.subheader("K-Means Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3, step=1)

    if st.sidebar.button("Run K-Means"):
        try:
            clustered_df, kmeans_model, feature_columns = kmeans_clustering(df, n_clusters)
            st.write("### K-Means Clustering Result")
            st.dataframe(clustered_df)

            # Show which records are in which cluster
            st.write("### Records in Each Cluster")
            cluster_groups = clustered_df.groupby('Cluster')
            for cluster_id, records in cluster_groups:
                st.write(f"#### Cluster {cluster_id}")
                st.dataframe(records)

            # Display centroids of the clusters
            centroids_df = pd.DataFrame(kmeans_model.cluster_centers_, columns=feature_columns)
            st.write("### Centroids of the Clusters")
            st.dataframe(centroids_df)

        except Exception as e:
            st.error(f"Error: {e}")


    elif algo_choice == "Decision Tree":
        st.sidebar.subheader("Decision Tree Parameters")
        target_column = st.sidebar.selectbox("Select Target Column", df.columns)
        max_depth = st.sidebar.slider("Max Depth of Tree", 1, 10, 3)

        if st.sidebar.button("Run Decision Tree"):
            try:
                clf = decision_tree(df, target_column, max_depth)

                # Decision Tree Visualization using Graphviz
                st.write("### Decision Tree Visualization")

                # Export decision tree to DOT format for graphviz
                dot_data = export_graphviz(
                    clf,
                    out_file=None,
                    feature_names=df.drop(target_column, axis=1).columns,
                    class_names=df[target_column].unique().astype(str),
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph)  # Display the decision tree plot in Streamlit

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.write("### Please upload a dataset to get started.")
