import codecs
import gzip
import numpy as np

class Glove:
    def __init__(self):
        self.wd_to_i = {}
        self.i_to_wd = {}
        self.v = None
    
    def load_glove(self, glove_path, gz=True, max_entries=20000, num_components=200):
        """Load GloVe vectors into memory from txt file. Returns a dict where the keys are 
        the headwords from the model and the values are the n-dimensional vectors
        representing the word.
    
        Arguments
        glove_path: path to glove vectors (txt file or gzipped txt file)
        gz: is path to a gzipped file (default True)"""

        self.v = np.empty((max_entries, num_components))
        if gz:
            with gzip.open(glove_path, "r") as glove_file:
                utfreader = codecs.getreader("utf-8")
                for i, glove_entry in enumerate(utfreader(glove_file)):
                    if i >= max_entries:
                        break
                    glove_components = glove_entry.split(' ')
                    self.wd_to_i[glove_components[0]] = i
                    self.i_to_wd[i] = glove_components[0]
                    self.v[i,:] = tuple(float(i) for i in glove_components[1:])
        else:
            with codecs.open(glove_path, "r", "utf-8") as glove_file:
                for i, glove_entry in enumerate(glove_file):
                    if i >= max_entries:
                        break
                    glove_components = glove_entry.split(' ')
                    self.wd_to_i[glove_components[0]] = i
                    self.i_to_wd[i] = glove_components[0]
                    self.v[i,:] = tuple(float(i) for i in glove_components[1:])

    
    def nearest_to_vec(self, vec, n=10):
        similarities = np.dot(self.v, vec) / (np.linalg.norm(self.v,axis=1) * 
                                                np.linalg.norm(vec))
        # sort similarities largest to smallest
        simil_i = np.argsort(-similarities)
        return [(self.i_to_wd[i], similarities[i]) for i in simil_i[:n]]
    
    def nearest_euclidean(self, vec, n=10):
        sse = np.sum((self.v-vec) ** 2, axis=1)
        distances = np.sqrt(sse)
        # sort distances smallest to largest
        simil_i = np.argsort(distances)
        return [(self.i_to_wd[i], distances[i]) for i in simil_i[:n]]
    
    def get_nearest(self, keyword, n=10):
        kw_vector = self.v[self.wd_to_i[keyword]]
        return self.nearest_to_vec(kw_vector, n)
    
    def plot_vec(self,vec):
        plt.bar(range(0, len(vec)), vec)

    def plot_wd(self,wd):
        self.plot_vec(self.v[self.wd_to_i[wd]])

    def get_vec(self,wd):
        return self.v[self.wd_to_i[wd]]
    
    def add_2_minus_1(self, to_add, to_subtract):
        return np.sum([self.get_vec(wd) for wd in to_add],axis=0) - self.get_vec(to_subtract)
    
    def __getitem__(self, key):
        # Pull some type awfulness
        if type(key) in (unicode, str):
            return self.get_vec(key)
        elif type(key) in (float, np.ndarray):
            return self.nearest_to_vec(key)
        else:
            raise IndexError()

