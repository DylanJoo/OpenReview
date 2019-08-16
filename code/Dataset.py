import pandas as pd
import numpy as np
import util

class icml_data():

    def __init__(self, yr = 19):
        self._load(yr)

    def _load(self, yr): #17-19
        a = 'ICML/ICML'+str(yr)+'_paper.csv'
        self.df = pd.read_csv(a)

    def getAbstract(self):
        return self._pp(self.df.Abstract).values

    def _pp(self, series):
        X = series.apply(lambda x: util.strnormalized(str(x)))
        X = X.apply(util.sent2words)
        #X = X.apply(util.swremoval)
        return X

class nips_data():

    def __init__(self, yr = 18):
        self._load(yr)

    def _load(self, yr): #13-18
        a = 'NIPS/NIPS'+str(yr)+'_paper.csv'
        self.df = pd.read_csv(a)
        
    def getAbstract(self):
        return self._pp(self.df.Abstract).values

    def getY(self):
        return self.df.Decision.values

    def _pp(self, series):
        X = series.apply(lambda x: util.strnormalized(str(x)))
        X = X.apply(util.sent2words)
        #X = X.apply(util.swremoval)
        return X
       

class iclr_data():

    def __init__(self, yr, select = 'submission'):
        self._load(yr, select)

    def _load(self, yr, select):
        
        if yr == 18:
            s = 'ICLR/ICLR18_submissions.csv'
            r = 'ICLR/ICLR18_reviews.csv'
        else:
            s = 'ICLR/ICLR17_submissions.csv'
            r = 'ICLR/ICLR17_reviews.csv'
        
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
        return self._pp(self.df.Review.apply(np.array)).values
    
    def getY(self):
        return self.df.Decision.values
    
    def getAbstract(self):
        return self._pp(self.df.Abstract).values

    def getTitle(self):
        return self._pp(self.df.Title).values

    def getRating(self):
        return self.df.Rating.apply(np.array).values

    def _pp(self, series):
        X = series.apply(lambda x: util.strnormalized(str(x)))
        X = X.apply(util.sent2words)
        #X = X.apply(util.swremoval)
        return X
    
class rt_data():

    def __init__(self):
        self.x = []
        self.y = []
        self._load()

    def _load(self):
        p = 'RT/rt_polarity_all.txt'
        data = open(p, encoding='utf-8', errors='ignore').readlines()

        for doc in data:
            y_, x_ = doc.split(' ', 1)

            self.x.append(self._pp(x_))
            self.y.append(int(y_))

    def getX(self):
        return self.x

    def getY(self):
        return self.y
    
    def _pp(self, x_):
        x = util.strnormalized(x_)
        x = util.sent2words(x)
        x = util.swremoval(x)
        return x
