import h5py


def repack(h5_file_path):
    """
    Repack the h5 file to reset disk usage.

    h5_file_path: str
        The path to the h5 file.

    """
    h5 = h5py.File(h5_file_path, "r")
    h5new = h5py.File(h5_file_path + "_temp", "w")
    for key, val in h5.items():
        h5.copy(key, h5new)
    for key, val in h5.attrs.items():
        h5new.attrs[key] = val
    h5.close()
    h5new.close()
    os.remove(h5_file_path)
    os.rename(h5_file_path + "_temp", h5_file_path)

def display_attrs(h5,pref=""):
    print(pref+"attrs:",end=" ")
    for key, val in h5.attrs.items():
        print("%s: %s" % (key, val),end="; ")
    print()

def display_recusive(h5,pref=""):
    for key in h5.keys():
        print(pref+key+":")
        if isinstance(h5[key], h5py.Dataset):
            print(pref+"  shape:",h5[key].shape)
        else:
            display_attrs(h5[key],pref+"  ")
            display_recusive(h5[key],pref+"  ")

def display_tree(h5_file_path):
    with h5py.File(h5_file_path, "r") as h5:
        display_recusive(h5)