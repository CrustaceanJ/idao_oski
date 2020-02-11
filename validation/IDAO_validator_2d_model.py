import numpy as np
import pandas as pd


class IDAO_validator_2d_model:
    """
    Валидация для 2d модели
    """
    
    def __init__(self, parts_amount, max_steps_amount):
        """
        parts_amount - 
        """
        
        self.parts_amount = parts_amount
        self.max_steps_amount = max_steps_amount
        
    def get_n_splits(self):
        return self.n_splits
    
    def split(self, X):
        """
        На выходе будет 2 двумерных numpy массив размера (sat_id, 2). Один для трейна, другой для теста.
        Каждая строка соответствует своему sat_id
        Значения 0 столбца - индекс начала трейна (теста) для данного sat_id
        Значения 1 столбца - индекс конца трейна (теста) для данного sat_id
        """
        
        for step in np.arange(1, self.max_steps_amount + 1):
            offset = 0
            train_idxs = np.zeros((X['sat_id'].nunique(), 2))
            test_idxs = np.zeros((X['sat_id'].nunique(), 2))
            if (step + 1) / self.parts_amount > 1.0:
                    print('WARNING: Не весь тест момещается -> пропускаем фолд')
                    continue
            for sat_id in X['sat_id'].unique():
                if sat_id != 0:
                    offset += len(X[X['sat_id'] == (sat_id - 1)])
                train_idxs[sat_id, 0] = offset
                train_idxs[sat_id, 1] = offset + int(step / self.parts_amount * X[X['sat_id'] == sat_id].shape[0])
                
                
                test_idxs[sat_id, 0] = offset + int(step / self.parts_amount * X[X['sat_id'] == sat_id].shape[0])
                test_idxs[sat_id, 1] = offset + int((step + 1) / self.parts_amount * X[X['sat_id'] == sat_id].shape[0])
                
            yield (train_idxs.astype(int), test_idxs.astype(int))
