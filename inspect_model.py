import h5py, json, sys
path='face_detection_model.h5'
try:
    with h5py.File(path,'r') as f:
        print('Root keys:', list(f.keys()))
        print('Attrs keys:', list(f.attrs.keys()))
        if 'model_config' in f.attrs:
            mc = f.attrs['model_config']
            print('\nmodel_config attr (first 2000 chars):')
            s = mc if isinstance(mc,str) else mc.decode('utf-8')
            print(s[:2000])
        elif 'model_config' in f:
            print('\nmodel_config dataset present')
            data = f['model_config'][()]
            s = data if isinstance(data,str) else data.decode('utf-8')
            print(s[:2000])
        else:
            if 'layer_names' in f:
                print('\nlayer_names count:', len(f['layer_names']))
            if 'model_weights' in f:
                print('model_weights keys:', list(f['model_weights'].keys())[:20])
except Exception as e:
    print('Error reading H5:', e)
    sys.exit(1)
