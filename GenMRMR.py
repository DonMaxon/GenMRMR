import numpy
import dask
import dask.array as da

from sklearn import feature_selection
from sklearn.metrics import f1_score
from ITMO_FS.filters.multivariate import MRMR
from dask import delayed
from sklearn.feature_selection import mutual_info_classif

class Family:
        
    """
    Pair of parents, it is used to create child

    Args:
        parent1, parent2(int): numbers of parents in population
    """

    parent1: int
    parent2: int

    def __init__(self, parent1, parent2):
        """
        Constructor of family

        Args:
            parent1, parent2(int): numbers of parents in population
        """
        self.parent1 = parent1
        self.parent2 = parent2

    def __eq__(self, family2):
        """
        It is used to compare 2 families, families are equal 
        if they have the same parents indepently from their number (parent1 or parent2)

        Args:
            family2: family to compare with
        """
        return (
            self.parent1 == family2.parent1
            and self.parent2 == family2.parent2
            or self.parent1 == family2.parent2
            and self.parent2 == family2.parent1
        )


class GenMRMR:

    """
    Class for GenMRMR feature selection algorithm. It is based on genetic algorithm where 
    probability of inserting a feature to resulting list is calculated using the MRMR algorithm.
    For more information visit https://github.com/DonMaxon/GenMRMR

    Args:
        classifier: classifier that will be used to evaluate individuals
        k: number of features to select, in the resulting dataset number of features will be less or equal k
        write_info: True if you want to see the whole information about process of feature selection
        __features: numbers of features that were selected to resulting dataset
    """
     
    classifier: object
    k: int
    write_info: bool
    __features = []

    def __init__(self, classifier, k: int, write_info=False):
        """
        Constructor of GenMRMR algorithm

        Args:
            classifier: classifier that will be used to evaluate individuals
            k: number of features to select, in the resulting dataset number of features will be less or equal k, k must be >= 0
            write_info: True if you want to see the whole information about process of feature selection, deafult value - False
        """
        self.classifier = classifier
        self.write_info=write_info
        if k<=0:
            raise ValueError("Количество признаков должно быть положительным")
        self.k = k
    

    def __sort_features(self, data: numpy.ndarray, labels: numpy.ndarray) -> dict:
        """
        Default feature sort to evaluate features. It is used to evaluate probability
        of adding features during mutation.

        Args:
            data: training data
            labels: labels for training data

        """
        selector = feature_selection.SelectKBest(score_func=mutual_info_classif, k='all')
        selector.fit(data, labels)
        ratings = {}
        for i in range(len(selector.scores_)):
            ratings[i]=selector.scores_[i]
        ratings = dict(sorted(ratings.items(), key=lambda item: item[1]))
        return ratings
    
    def __estimate_individuum(self, individuum: numpy.ndarray, x_train: numpy.ndarray, y_train: numpy.ndarray, x_cv: numpy.ndarray, y_cv: numpy.ndarray) -> float:
        """
        Method for estimating individuums. It is used to evaluate probability
        of using this individual during crossover. Also it is used to understand have we got a stagnation.

        Args:
            individuum: in fact, array of features
            x_train, y_train, x_cv, y_cv: data and labels that will be used by classifier to evaluate individual quality

        """        
        new_x_cv = numpy.empty((numpy.shape(x_cv)[0], 0), dtype=float)
        new_x_train =  numpy.empty((numpy.shape(x_train)[0], 0), dtype=float)
        for i in range(len(individuum)):
            new_x_train = numpy.column_stack([new_x_train, numpy.array(x_train[:, individuum[i]])])
            new_x_cv = numpy.column_stack([new_x_cv, numpy.array(x_cv[:, individuum[i]])])
        self.classifier.fit(new_x_train, y_train)
        y_cv_res = self.classifier.predict(new_x_cv)
        return f1_score(y_cv_res, y_cv, average='weighted')
    
    def __ratings_in_population(self, population: numpy.ndarray, ratings: numpy.ndarray) -> dict:
        """
        Method is used to bind individuals and their quality

        Args:
            population: array of individuals
            ratings: array of ratings of individuals

        """  
        ratings_of_individuums={}
        for i in range(len(population)):
            ratings_of_individuums[i]=ratings[i]
        ratings_of_individuums=dict(sorted(ratings_of_individuums.items(), key=lambda item: item[1]))
        return ratings_of_individuums
    
    def __deviation_from_max(self, qualities: numpy.ndarray) -> float:
        """
        Method is used get deviation from the best result in array of qualities

        Args:
            qualities: array of qualities of individuals or features
        """  
        m = max(qualities)
        res = 0
        for i in range(len(qualities)):
            res+=(qualities[i]-m)**2
        res/=len(qualities)
        return res
    
    def __find_parent(self, parent_data: float, qualities: numpy.ndarray, ratings_in_population: dict, partner=None, parents = []) -> numpy.ndarray:
        """
        Method is used to find parent to create a family. 
        Parent is chosen using parent_data that was generated randomly.
        Parent shouldn't be equal to its partner in family.
        Parent shouldn't create a family with a partner that is equal to any other family in population

        Args:
            parent_data: required quality of parent
            qualities: array of qualities of parents
            ratings_in_population: binding for numbers of individuals and their qualities
            patner: parent1 from Family if it was chosen before
            parents: all families that were created before in this iteration
        """  
        if parent_data>max(qualities):
            parent_data=max(qualities)-(parent_data-max(qualities))
        prev_individuum = None
        for index, (individuum, quality) in enumerate(ratings_in_population.items()):
            if quality>parent_data:
                if partner!=individuum and Family(individuum, partner) not in parents:
                    return individuum                    
            if partner!=individuum and Family(individuum, partner) not in parents:
                prev_individuum = individuum
        if prev_individuum==None:
            for index, (individuum, quality) in enumerate(ratings_in_population.items()):
                if partner!=individuum and Family(individuum, partner) not in parents:
                    return individuum
                if individuum!=partner:
                    prev_individuum=individuum
        return prev_individuum
    

    def __generate_family(self, qualities: numpy.ndarray, ratings_in_population: dict, variance: float, parents) -> Family:
        """
        Method is used to generate a new family

        Args:
            qualities: array of qualities of parents
            ratings_in_population: binding for numbers of individuals and their qualities
            variance: deviation that was calculated in __deviation_from_max method
            parents: all families that were created before in this iteration
        """  
        new_family_data = numpy.random.normal(max(qualities), numpy.sqrt(variance), 2)
        parent1 = self.__find_parent(new_family_data[0], qualities, ratings_in_population)
        parent2 =  self.__find_parent(new_family_data[1], qualities, ratings_in_population, parent1, parents)
        return Family(parent1, parent2)
    

    def __create_pairs_of_parents(self, ratings_in_population: dict, qualities: list, deviation: float) -> list:
        """
        Method is used to population

        Args:
            qualities: array of qualities of parents
            ratings_in_population: binding for numbers of individuals and their qualities
            deviation: deviation that was calculated in __deviation_from_max method
        """  
        parents = []
        for i in range(len(qualities)):
            new_family = self.__generate_family(qualities, ratings_in_population, deviation, parents)
            parents.append(new_family)
        return parents
    
    def __find_parents_similarity(self, parents: Family, population: numpy.ndarray) -> float:
        """
        Method is used find parents similarity and quality to evaluate probability of mutation

        Args:
            parents: parents from this Family will be evaluated the more similarity they have the more probability of mutation
            population: population of individuals
        """  
        res = 0
        for i in range(len(population[0])):
            if population[parents.parent1][i] in population[parents.parent2]:
                res+=1
        return res/(2*len(population[0]))
    
    def __create_children(self, parents: list, population: numpy.ndarray, qualities: numpy.ndarray, prev_f: float, deviation: float, ratings_of_features: dict, x_train: numpy.ndarray, y_train: numpy.ndarray) -> numpy.ndarray:
        """
        Method is used to create next generation for genetic algorithm.
        If parents have a big part of equal features they have the more probability of mutation.
        More difference between weighted F1-score between Family memebers and best individual they have the more probability of mutation.
        Adding features to child will perform simultaneously if algorithm works on machine that has several cores.

        Args:
            parents: list of all families
            population: array of all individuals
            qualities: array of qualities for each individual
            prev_f: weghted F1-score of best ever before individual
            deviation: deviation that was calculated in __deviation_from_max method
            ratings_of_features: ratings of features to check for required quality
            x_train, y_train: train data and labels to add features to child using MRMR
        """  
        npop = []
        for i in range(len(parents)):
            #Если качество предыдущего поколения больше, то вероятность мутации больше
            if (prev_f-qualities[i]<0):
                mutation_probability = self.__find_parents_similarity(parents[i], population)+max(qualities)-qualities[i]
            else:
                mutation_probability = self.__find_parents_similarity(parents[i], population) + max(qualities)+prev_f - 2*qualities[i]
            if mutation_probability<0:
                mutation_probability=0
            if mutation_probability>1:
                mutation_probability=1
            number_of_new_features = 1+(int)(mutation_probability*len(population[0])-1)
            new_features = self.__generate_new_features(parents[i], population, list(ratings_of_features.values()), number_of_new_features, deviation, ratings_of_features)
            parent1 = population[parents[i].parent1]
            parent2 = population[parents[i].parent2]
            parent1 = numpy.setdiff1d(parent1, new_features)
            parent2 = numpy.setdiff1d(parent2, new_features)
            features_of_parents=numpy.concatenate((parent1, parent2))
            npop.append(delayed(self.__add_features_to_child)(population, number_of_new_features, features_of_parents, new_features, x_train, y_train, parent1, parent2))
        npop = dask.compute(*npop)
        return numpy.array(npop)
    
    def __create_population(self, num_of_features: int) ->numpy.ndarray:
        """
        Method is used create initial population

        Args:
            num_of_features: number of features to select
        """  
        individuums = numpy.random.randint(num_of_features, size = (num_of_features, num_of_features))
        return individuums
    
    def __add_features_to_child(self, population, number_of_new_features, features_of_parents, new_features, x_train, y_train, parent1, parent2):
        """
        Method is used create initial population

        Args:
            num_of_features: number of features to select
        """  
        for j in range(len(population[0]) - number_of_new_features):
            mrmr_scores = MRMR(new_features, features_of_parents, x_train, y_train)
            if len(mrmr_scores)>len(features_of_parents):
                mrmr_scores = [mrmr_scores[i] for i in features_of_parents]
            new_features = numpy.resize(new_features, len(new_features) + 1)
            max_score = float('-inf')
            ind = -1

            for i in range(len(mrmr_scores)):
                if mrmr_scores[i] > max_score and features_of_parents[i] not in new_features:
                    ind = i
                    max_score = mrmr_scores[i]
            new_feature = ind
            if new_feature < len(parent1):
                new_features[-1] = parent1[new_feature]
            else:
                new_features[-1] = parent2[new_feature - len(parent1)]
        return new_features
    
    def __find_feature_for_mutation(self, feature_min_quality: float, qualities: numpy.ndarray, ratings_of_features: list, selected: numpy.ndarray, features_of_parents: numpy.ndarray):
        """
        Method is used to find feature to add to child during mutation.

        Args:
            qualities: array of qualities of parents
            feature_min_quality: required quality for feature to add it
            ratings_of_features: ratings of features to check for required quality
            selected: features that were already selected to individual
            features_of_parents: features that are used in parents
        """  
        if feature_min_quality>max(qualities):
            feature_min_quality=max(qualities)-(feature_min_quality-max(qualities))
        prev_feature = None
        for index, (feature, quality) in enumerate(ratings_of_features.items()):
            if quality>feature_min_quality:
                if feature in selected or feature in features_of_parents:
                    if index==len(ratings_of_features)-1:
                        return prev_feature
                else:
                    return feature
            if feature in selected or feature in features_of_parents:
                prev_feature = feature
        if prev_feature==None:
            for index, (feature, quality) in enumerate(ratings_of_features.items()):
                    if feature not in selected and feature not in features_of_parents:
                        return feature
                    if feature not in selected:
                        prev_feature=feature
        return prev_feature

    def __generate_new_features(self, parents: Family, population: numpy.ndarray, qualities: numpy.ndarray, number_of_new_features: int, deviation: float, ratings_of_features:dict):
        """
        Method is used to generate features during mutation and adding them to chils.

        Args:
            qualities: array of qualities of parents
            parents: family that creates a child
            population: population of individuals
            number_of_new_features: number of features that will be added during mutation
            deviation: deviation that was calculated in __deviation_from_max method
            ratings_of_features: ratings of features to check for required quality
        """  
        res = numpy.full(shape=(number_of_new_features), fill_value=-1, dtype=int)
        for i in range(len(res)):
            new_feature_data = numpy.random.normal(max(qualities), numpy.sqrt(deviation), 1)
            features_of_parents = numpy.vstack((population[parents.parent1], population[parents.parent2]))
            new_feature = self.__find_feature_for_mutation(new_feature_data[0], qualities, ratings_of_features, res, features_of_parents)
            res[i]=new_feature
        return res
    
    def __check_datasets(self, x_train: numpy.ndarray, x_cv: numpy.ndarray):
        """
        Method is used to check datasets on constraints

        Args:
            x_train, x_cv: train and cv data that will be used for feature selection
        """  
        if x_train.shape[1]<=self.k or x_cv.shape[1]<=self.k:
            raise ValueError("Number of features to select should be less or equal than number of features in dataset")
        if x_train.shape[1]<3 or x_cv.shape[1]<3:
            raise ValueError("Number of features is less or equal than 3")
        if self.k<2:
            raise ValueError("k must be more than 1")
        if x_cv.shape[1]!=x_train.shape[1]:
            raise ValueError("Number of features in cv and train datasets must be equal")
        pass

    def fit(self, x_train, y_train, x_cv, y_cv):
        """
        Method is used to fit GenMRMR to given

        Args:
            x_train, y_train, x_cv, y_cv: train and cv data and labels that will be used for feature selection
        """  
        self.__check_datasets(x_train, x_cv)
        population = self.__create_population(self.k)
        ratings_of_features = self.__sort_features(x_train, y_train)
        prev_best_F_weighted = 0
        population_best_F = 0.00
        best_individuum = 0
        iterations_without_increase_f = 0
        while population_best_F-prev_best_F_weighted>0.0001 or iterations_without_increase_f<=2:
            iterations_without_increase_f+=1
            qualities = []
            prev_best_F_weighted=population_best_F
            if (self.write_info):
                print("Estimating individuums")
            for i in range(len(population)):
                cur_f_of_individuum  = self.__estimate_individuum(population[i], x_train, y_train, x_cv, y_cv)
                if (population_best_F<cur_f_of_individuum):
                    population_best_F=cur_f_of_individuum
                    best_individuum=population[i]
                    iterations_without_increase_f=0
                qualities.append(cur_f_of_individuum)
                #print(cur_f_of_individuum)
            deviation_of_qualities = self.__deviation_from_max(qualities)
            ratings_of_individuums = self.__ratings_in_population(population, qualities)
            if (self.write_info):
                print("Creating parents")
            parents = self.__create_pairs_of_parents(ratings_of_individuums, qualities, deviation_of_qualities)
            deviation_of_features=self.__deviation_from_max(list(ratings_of_features.values()))
            if (self.write_info):
                print("Creating children")
            new_population = self.__create_children(parents, population, qualities, prev_best_F_weighted, deviation_of_features, ratings_of_features, x_train, y_train)
            population=new_population
        self.__features = numpy.unique(best_individuum)

    def transform(self, data: numpy.ndarray):
        """
        Method is used to transform fitted GenMRMR

        Args:
            data: data to transform
        """  
        if (self.__features==[]):
            raise ValueError("GenMRMR is not fitted")
        result = data[:, self.__features[0]]
        for i in range(1, len(self.__features)):
            result = numpy.column_stack((result, data[:, self.__features[i]]))
        return result
    