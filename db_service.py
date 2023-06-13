import pymongo
import gridfs
import pickle
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
    db.feature_names.insert_one({"SCE__feature_names": load_feature_names(1).tolist()})
    db.feature_names.insert_one(
        {"Haifa__feature_names": load_feature_names(2).tolist()}
    )


def get_feature_names(option):
    if option == 1:
        return db.feature_names.find_one("SCE__feature_names")
    else:
        return None
