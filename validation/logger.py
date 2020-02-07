import pandas as pd
import os

class Logger:
    def __init__(self, logsFileName, mode):
        """
        :param logsFileName: Name of dataframe file or txt
        :param mode: 'df' for DataFrame
                     'txt' for txt file
        """

        self.logsDir = './Logs/'
        self.logsFileName = logsFileName
        self.mode = mode

        # Open existed file or create DataFrame

        if os.path.exists(self.logsDir + self.logsFileName):
            self.open()
        else:
            if mode == 'df':
                self.logsFile = pd.DataFrame()
            elif mode == 'txt':
                self.open()
            else:
                print("Incorrect mode! Available modes: 'df', 'txt'")

    def open(self):
        if self.mode == 'df':
            self.logsFile = pd.read_csv(self.logsDir + self.logsFileName)
        elif self.mode == 'txt':
            self.logsFile = open(self.logsDir + self.logsFileName, 'a+')
        else:
            print("Incorrect mode! Available modes: 'df', 'txt'")


    def save(self):
        if self.mode == 'df':
            self.logsFile.to_csv(self.logsDir + self.logsFileName, index=False)
        elif self.mode == 'txt':
            self.logsFile.close()
        else:
            print("Incorrect mode! Available modes: 'df', 'txt'")

    def add_empty_row(self):
        if self.mode == 'df':
            self.logsFile = self.logsFile.append(pd.Series(), ignore_index=True)
        elif self.mode == 'txt':
            self.logsFile.write("-"*40 + "\n"*2)
        else:
            print("Incorrect mode! Available modes: 'df', 'txt'")

    def add_data(self, column_name, data):
        if self.mode == 'df':
            self.logsFile.loc[self.logsFile.index[-1], column_name] = data
        elif self.mode == 'txt':
            if column_name is not None:
                self.logsFile.write(column_name + ': ' + str(data) + '\n')
            else:
                self.logsFile.write(str(data) + '\n')
        else:
            print("Incorrect mode! Available modes: 'df', 'txt'")
