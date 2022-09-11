import h5py
import io
import numpy as np


class H5Helper:
    """Some helper function related to h5py."""
    h5_element_type = \
        (np.int64, np.float64, str, np.float, float, np.float32, int)
    h5_list_type = \
        (list, np.ndarray)

    @staticmethod
    def save_attrs_to_h5file(h5file: h5py.File,
                             root_key: str = '/',
                             dic: dict = {}):
        for k, v in dic.items():
            h5file[root_key].attrs[k] = v
        return None

    @staticmethod
    def recursively_save_dict_contents_to_h5file(h5file: h5py.File,
                                                 root_key: str = '/',
                                                 dic: dict = {}):
        if not root_key.endswith('/'):
            root_key = root_key + '/'
        for k, v in dic.items():
            k = str(k)
            if k == 'attrs':
                if root_key not in h5file:
                    h5file.create_group(root_key)
                H5Helper.save_attrs_to_h5file(h5file, root_key, v)
                continue
            if isinstance(v, dict):
                H5Helper.recursively_save_dict_contents_to_h5file(
                    h5file, root_key + k + '/', v)
            elif isinstance(v, H5Helper.h5_element_type):
                h5file[root_key + k] = v
            elif isinstance(v, H5Helper.h5_list_type):
                try:
                    h5file[root_key + k] = v
                except TypeError:
                    v = np.array(v).astype('|S9')
                    h5file[root_key + k] = v
            else:
                raise TypeError(f'Cannot save {type(v)} type.')
        return None

    @staticmethod
    def load_dict_from_hdf5(filename):
        with h5py.File(filename, 'r') as h5file:
            return H5Helper.recursively_load_dict_contents_from_group(
                h5file, '/')

    @staticmethod
    def load_h5group_attr_to_dict(h5file: h5py.File,
                                  root_key: str = '/') -> dict:
        ret_dict = {}
        for k, v in h5file[root_key].attrs.items():
            ret_dict[k] = v
        return ret_dict

    @staticmethod
    def recursively_load_dict_contents_from_group(h5file: h5py.File,
                                                  root_key: str = '/'):
        if not root_key.endswith('/'):
            root_key = root_key + '/'
        ret_dict = {}
        if len(h5file[root_key].attrs) > 0:
            ret_dict['attrs'] = H5Helper.load_h5group_attr_to_dict(
                h5file, root_key)
        for key, item in h5file[root_key].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ret_dict[key] = item[...]
            elif isinstance(item, h5py._hl.group.Group):
                ret_dict[key] = \
                    H5Helper.recursively_load_dict_contents_from_group(
                    h5file, root_key + key + '/')
        return ret_dict

    @staticmethod
    def h5py_to_binary(h5f):
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as biof:
            for key, value in h5f.items():
                h5f.copy(
                    value,
                    biof,
                    expand_soft=True,
                    expand_external=True,
                    expand_refs=True)
        return bio
