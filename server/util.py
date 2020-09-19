import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,area,bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = area
    x[1] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0])

def get_location_name():
    global __locations

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[2:]

    return __locations


def load_saved_artifacts():
    print('loading saved artifacts...start')
    global __data_columns
    global __locations
    global __model

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[2:]

    with open('./artifacts/mumbai_home_price_model.pickle', 'rb') as f:
        __model = pickle.load(f)

    print('loading saved artifacts...done')


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_name())
    print(get_estimated_price('Andheri', 600, 1))
    print(get_estimated_price('Kharghar', 1000, 2))
    print(get_estimated_price('4 bunglows', 600, 1))
    print(get_estimated_price('Andheri', 600, 2))





