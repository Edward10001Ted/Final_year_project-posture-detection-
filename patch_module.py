import h5py
p='face_detection_model.h5'
with h5py.File(p,'r+') as f:
    s = f.attrs['model_config']
    s = s if isinstance(s,str) else s.decode('utf-8')
    if '"module": "builtins"' in s:
        s2 = s.replace('"module": "builtins"', '"module": "face_ver"')
        f.attrs['model_config'] = s2
        print('Rewrote module from builtins to face_ver')
    else:
        print('No builtins module found')
