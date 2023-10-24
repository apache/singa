import json
import pickle
import os.path


def read_json(file_name):
    print(f"Loading {file_name}...")
    is_exist = os.path.exists(file_name)
    if is_exist:
        with open(file_name, 'r') as readfile:
            data = json.load(readfile)
        return data
    else:
        print(f"{file_name} is not exist")
        return {}


def write_json(file_name, data):
    print(f"writting {file_name}...")
    with open(file_name, 'w') as outfile:
        outfile.write(json.dumps(data))


def read_pickle(file_name):
    print(f"Loading pickel {file_name}...")
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(file_name, data):
    print(f"writing pickle {file_name}...")
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    a = {1:1}
    write_json("./asdf.json", a)
    b = {2:2323}
    write_json("./asdf.json", b)

