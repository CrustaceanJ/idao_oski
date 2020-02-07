import ipyvolume as ipv
import ipyvolume.pylab as p3

import numpy as np

from ipywidgets import FloatSlider, ColorPicker, VBox, jslink


class processVisualizer:
    def __init__(self, train_data, test_data):
        """
            train_data - pandas DataFrame with train data
            test_data - pandas DataFrame with test
            
        """
        self.train_data = train_data
        self.test_data = test_data
        self.unique_sat_ids_amount = len(set(train_data['sat_id'].values).union(set(test_data['sat_id'].values)))
    
    def _prepare_stream(self, mode='train', obj_idxs=None,
                        max_time_steps=None, sim=None):
        """
             Вспомогательная функция. Её не нужно будет использовать, поэтому можете не вникать :))
        """
        if mode=='train':
            data = self.train_data
        else:
            data = self.test_data

        if sim:
            cols = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
        else:
            cols = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
            
        stream = np.zeros((6, max_time_steps, self.unique_sat_ids_amount))
                                     
        for sat_id in obj_idxs:
            sat_id_arr = data[data['sat_id'] == sat_id][cols].values.T
            time_steps_amount = sat_id_arr.shape[1]
            if time_steps_amount < max_time_steps:
                sat_id_arr = np.hstack((sat_id_arr, np.zeros((6, max_time_steps - time_steps_amount))))
            for i in range(6):
                stream[i, :, sat_id] = sat_id_arr[i]
        return stream
        
    
    def visualize_objects(self, train_idxs=None, test_idxs=None,
                          max_time_steps=None,
                          train_sim=False, test_sim=True,
                          train_marker_size=4, test_marker_size=6):
        """
            train_idxs - numpy array из индексов объектов (sat_id) тренировочной выборки,
                         которые надо визуализировать. Если  None - берем все объекты
                         
            test_idxs - numpy array из индексов объектов (sat_id) тренировочной выборки,
                         которые надо визуализировать. Если None - берем train_idxs
                         
            max_time_steps - максимальное количество измерений для одного объекта (sat_id)
            
            train_sim - если False - используем реальные данные (колонки без приставки sim)
            
            test_sim - если False - используем реальные (предсказанные) данные 
            (для этого в датафрейм нужно добавить приставки колонки с предсказаниями без приставки sim,
             как в трейне)
            
        """
        
        ipv.clear()
        if train_idxs is None: 
            train_idxs = np.array(self.train_data['sat_id'].unique())
        if test_idxs is None:
            test_idxs = train_idxs
            
        if max_time_steps is None:
            max_time_steps_train = self.train_data.groupby('sat_id').count()['epoch'].max()
            max_time_steps_test = self.test_data.groupby('sat_id').count()['epoch'].max()
            max_time_steps = max(max_time_steps_train, max_time_steps_test)
                 
        ## подготовка трейна и теста
        stream_train = self._prepare_stream('train', train_idxs, max_time_steps, train_sim)
        stream_test = self._prepare_stream('test', test_idxs, max_time_steps, test_sim)
        
        ## визуализация
        stream = np.dstack([stream_train[:, :, :], stream_test[:, :, :]])
        selected = stream_train.shape[2] + test_idxs
        self.q = ipv.quiver(*stream[:, :, :],
                           color="green", color_selected='red',
                           size=train_marker_size, size_selected=test_marker_size,
                           selected=selected)
        
        ##  Чтобы можно было менять размеры и цвета
        size = FloatSlider(min=1, max=15, step=0.2)
        size_selected = FloatSlider(min=1, max=15, step=0.2)
        color = ColorPicker()
        color_selected = ColorPicker()
        jslink((self.q, 'size'), (size, 'value'))
        jslink((self.q, 'size_selected'), (size_selected, 'value'))
        jslink((self.q, 'color'), (color, 'value'))
        jslink((self.q, 'color_selected'), (color_selected, 'value'))
#         ipv.style.use('seaborn-darkgrid')
        ipv.animation_control(self.q, interval=75)
        ipv.show([VBox([ipv.gcc(), size, size_selected, color, color_selected])])
#         ipv.show()
