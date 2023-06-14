import pymongo
import gridfs
import pickle
import streamlit as st
from functions import *


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    db_uri = (
        "mongodb+srv://"
        + st.secrets["username"]
        + ":"
        + st.secrets["password"]
        + "@cluster0.07cutbq.mongodb.net/?retryWrites=true&w=majority"
    )
    return pymongo.MongoClient(db_uri)


client = init_connection()
db = client.PVC
fs = gridfs.GridFS(db)


def set_datasets():
    db.datasets.insert_one(
        {"SCE__dataset": read_dataset1().to_dict("list"), "dataset_number": 1}
    )
    db.datasets.insert_one(
        {
            "Haifa__dataset": read_dataset2().to_dict("list"),
            "dataset_number": 2,
        }
    )


def set_models():
    sce_model = pickle.dumps(load_model(1, "svm_model_1.pickle"))
    fs.put(sce_model, filename="SCE__model")
    haifa_model = pickle.dumps(load_model(2, "svm_model_3.pickle"))
    fs.put(haifa_model, filename="Haifa__model")


def set_feature_names():
    db.feature_names.insert_one(
        {"name": "SCE__feature_names", "data": load_feature_names(1).tolist()}
    )
    db.feature_names.insert_one(
        {"name": "Haifa__feature_names", "data": load_feature_names(2).tolist()}
    )


def set_contributions(dataset):
    dataset = pd.read_csv(dataset, encoding="ISO-8859-1")
    db.contributions.insert_one({"dataset": dataset.to_dict("list")})


@st.cache_data()
def get_feature_names(option):
    if option == 1:
        return db.feature_names.find_one({"name": "SCE__feature_names"})["data"]
    if option == 2:
        return db.feature_names.find_one({"name": "Haifa__feature_names"})["data"]
    else:
        return None


@st.cache_data()
def get_model(option):
    if option == 1:
        return pickle.loads(fs.find_one({"filename": "SCE__model"}).read())
    if option == 2:
        return pickle.loads(fs.find_one({"filename": "Haifa__model"}).read())
    else:
        return None


@st.cache_data()
def get_dataset(option):
    if option in [1, 2]:
        name = "SCE" if option == 1 else "Haifa"
        return pd.DataFrame.from_dict(
            db.datasets.find_one({"dataset_number": option})[f"{name}__dataset"],
            orient="index",
        )
    else:
        return None
