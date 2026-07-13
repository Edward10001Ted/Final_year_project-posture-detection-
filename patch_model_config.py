import h5py, shutil, sys
src='face_detection_model.h5'
bak='face_detection_model.h5.bak'
print('Backing up', src, '->', bak)
shutil.copyfile(src,bak)
with h5py.File(src, 'r+') as f:
    if 'model_config' in f.attrs:
        mc = f.attrs['model_config']
        s = mc if isinstance(mc,str) else mc.decode('utf-8')
        if '__ellipsis__' in s:
            print('Found __ellipsis__ in model_config, patching...')
            s2 = s.replace('"class_name": "__ellipsis__"', '"class_name": "Ellipsis", "module": "builtins"')
            # write back
            f.attrs['model_config'] = s2
            print('Patched model_config in HDF5')
        else:
            print('__ellipsis__ not found in model_config')
    else:
        print('model_config not in H5 attrs')
print('Done')
