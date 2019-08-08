import pandas as pd
import numpy as np
import util

class or_data():

    def __init__(self, yr, select = 'submission'):
        self._load(yr, select)

    def _load(self, yr, select):
        
        if yr == 18:
            s = 'ICLR18/ICLR18_submissions.csv'
            r = 'ICLR18/ICLR18_reviews.csv'
        else:
            s = 'ICLR17/ICLR17_submissions.csv'
            r = 'ICLR17/ICLR17_reviews.csv'
        
        if select == 'submission':
            self.df = pd.read_csv(s)
            self.df.Decision.replace(to_replace = 2, value = 1, inplace = True)
            self.df.Decision.replace(to_replace = -1, value = 0, inplace = True)
            
        elif select == 'review':
            df_r = pd.read_csv(r).groupby('PID', as_index = False).agg(lambda x: x.tolist())
            self.df = pd.read_csv(s).join(df_r.set_index('PID'), \
                                          on = 'PID', lsuffix='_paper', rsuffix='_review')
            self.df.Decision.replace(to_replace = 2, value = 1, inplace = True)
            self.df.Decision.replace(to_replace = -1, value = 0, inplace = True)


    def getReview(self):
        return [doc for doc in self._pp(self.df.Review)]
    
    def getY(self):
        return list(self.df.Decision)
    
    def getAbstract(self):
        return [doc for doc in self._pp(self.df.Abstract)]

    def getTitle(self):
        return [doc for doc in self._pp(self.df.Title)]

    def getRatin(self):
        return list(self.df.Rating)

    def _pp(self, series):
        import util
        X = series.apply(lambda x: util.strnormalized(str(x)))
        X = X.apply(util.sent2words)
        X = X.apply(util.swremoval)
        return X
    
