
import matplotlib; matplotlib.use("agg")


import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
import numpy as np



def net():
    l_in = InputLayer((None, 16, 28, 28))
    l_conv = Conv2DLayer(l_in, num_filters=512, filter_size=5)
    return l_conv



l_out = net()



X = T.tensor4('X')
net_out = get_output(l_out, X)



get_out = theano.function([X], net_out)



fake_data = np.random.normal(0, 1, size=(1, 16, 28, 28))



latent_rep = get_out(fake_data)[0]
latent_rep.shape



latent_vectors=[]
for i in range(0, latent_rep.shape[1]):
    for j in range(0, latent_rep.shape[2]):
        #print latent_rep[:,i,j].shape
        latent_vectors.append(latent_rep[:,i,j])



latent_vectors = np.asarray(latent_vectors)
latent_vectors.shape


# We now have 24*24=576 vectors of size 512


from sklearn.manifold import TSNE



model = TSNE(n_components=20, random_state=0)



model.fit_transform(latent_vectors+0.01)



np.sum(np.isinf(latent_vectors))



np.isnan(latent_vectors).any()





