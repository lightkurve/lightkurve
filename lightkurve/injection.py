class UniformDistibution(object):
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def sample(self, size=1):
        return np.random.uniform(self.lb, self.ub, size)

    def plot()

    def __repr__():


class GaussianDistribution(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def sample(self, size=1):
        return np.random.normal(self.mean, self.var, size)

    def plot()

    def __repr__():

class TransitModel(......):

    def __init__(self,t0=UniformDistribution(0,100).
                     period=GaussianDistribution(1.1,0),
                     rprs=0.01):
        self.t0 = t0
        self.multiplicative = True

    def evaluate(time, **params):

        period = self.period.sample()
        return flux

class SupernovaModel(.....):
    '''
    '''
    def __init__(self,t0=UniformDistribution...Source, z, ):
        self.t0 = t0
        self.multiplicative = False

    def evaluate(time, **params):
        t0 = self.t0.sample()

        return flux

def inject(lc, model, *params):
    if model.multiplicative is True:
        return lc * model.evaluate(*params)
    else:
        return lc + model.evaluate(*params)
