
# GenMRMR

This repository proposes a new feature selection algorithm based on genetic and the MRMR algorithms. The algorithm aims to select a subset of features that maximizes classification accuracy. The algorithm works by creating a population of subsets of features, evaluating their weighted F1-score, and then hybridizing the best individuals to create new subsets. The probability of adding a feature during hybridization is estimated using the MRMR algorithm. Finally, the best subset of features is selected based on the F1-score. 

## Documentation

There are 3 methods available: fit, transform, fit_transform: 

fit method is used to fit GenMRMR to given

    Args:
        x_train, y_train, x_cv, y_cv(numpy.ndarray): train and cv data and labels that will be used for feature selection

transform method is used to transform fitted GenMRMR

    Args:
        data(numpy.ndarray): data to transform

fit_transform method is used to fit and then transform data using GenMRMR

    Args:
        data(pandas.core.frame.DataFrame): data to transform
        labels(pandas.core.series.Series): labels for data



## Authors

- Algorithm created by Maksim Tislenko(makstislenko@gmail.com)
    - Created the idea of algorithm
    - Implemented the whole algorithm
- Сomparison with existing feature selection algorithms was conducted with the participation of Rustam Paringer (rusparinger@ssau.ru) from the Department of Technical Cybernetics at Samara University.


## Acknowledgements

I would also like to express my gratitude to Arkadiy Sotnikov (cotnikoarkady@gmail.com) for assistance in preparing this readme.

