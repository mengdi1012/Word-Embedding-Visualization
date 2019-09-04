import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from scipy import spatial
 
 
def main():    
    embeddings_file = 'data/word_embeddings.p'
    embed_dict = load_embeddings(embeddings_file)
    
    vocabulary = embed_dict.keys()
    word_vec = np.array(embed_dict.values())

    ############################################################################
    # You should modify this part by selecting a subset of word embeddings 
    # for better visualization
    ############################################################################
    
    for i in range(100, 800):
        if(vocabulary[i] == 'uniforms'):
            print i
            print 'herer'

    #glasses 493
    uniforms_vec = word_vec[493, :]
    sim_far = []
    sim_close = []
    for i in range(200, 600):
        sim = 1- spatial.distance.cosine(uniforms_vec, word_vec[i])
        if sim > 0.75:
            sim_close.append(vocabulary[i])
        if sim < 0.25:
            sim_far.append(vocabulary[i])
    sim_far = sim_far[0:10]
    sim_close = sim_close[0:10]
    print 'Far---------------------'
    for i in range(10):
        print sim_far[i]
    print '\nClose--------------------'
    for i in range(10):
        print sim_close[i]
            


    word_vec = word_vec[200:600, :]    
    vocabulary = vocabulary[200:600]



    ############################################################################

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)    
    Y = tsne.fit_transform(word_vec)
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def load_embeddings(file_name):
    """ Load in the embeddings """
    return pickle.load(open(file_name, 'rb'))


if __name__ == '__main__':    
    main()
