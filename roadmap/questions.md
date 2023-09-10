- Which hyperparameters should be optimized?
    
    Currently, the CatBoost based models allow the user to optimize the hyperparameters listed in `optimization_candidate_init_args` that can be found [here](https://github.com/0xideas/100gecs/blob/ffcae80e2bdd410cdd1d00d0382685db22210762/src/gecs/catgec.py#L261) `categorical_hyperparameters` just below.
    And the LGBM based models allow the uer to optimize the hyperparameters listed with the same variable names from [here](https://github.com/0xideas/100gecs/blob/ffcae80e2bdd410cdd1d00d0382685db22210762/src/gecs/lightgec.py#L231)
    Are these the right hyperparameters to tune? Are there any important ones missing?


- What are the appropriate hyperparameter ranges and intervals?
    
    The current sampling ranges for each hyperparameter can be found [here](https://github.com/0xideas/100gecs/blob/ffcae80e2bdd410cdd1d00d0382685db22210762/src/gecs/gec_base.py#L108). Are they the right ones?


- What kernel should the gaussian process use, and how should it be parametrized?
    
    Currently, an `RBF` kernel with `l=1.0` is used, to model hyperparameters projected to a ((-1.0, 1.0),..., (-1.0, 1.0)) space. 


- What is the optimal acquisition function over gaussian process outputs?
    
    This is currently 0.7. How does this interact with grid position pre-filtering?


- Are there good alternative models to model the joint distribution of scores over hyperparameter values?
    
    Currently this is a gaussian process. Is there some other model that might perform better?


- How many random rounds should be done by default?
    
    Each GEC starts with a fixed number of random hyperparameter samples. Currently, they will be sampled randomly 5 or 0.5*n_iter times, whichever is lower. Should that constant be increased?


- How many grid positions should be evaluated using the gaussian process?
    
    At each iteration, a sample of 1000 grid positions in the hyperparameter space is taken and the conditional probability distribution of the score is predicted using the gaussian process. Should this evaluation done for more positions, or fewer?


- Should there be a pre-filtering of grid positions based on proximity to known grid positions, and if yes, what ratio of initial grid positions should be used?
    
    The GEC implements the capability to score the sampled grid positions according to their proximity to the best evaluated grid positions, and only take the closest X% for evaluation. Should this be used, and if yes, what share should be taken?


- What distance metric should be used for pre-filtering?
    
    The pre-filtering is based on some distance metric. This is currently set to `cityblock` What is the appropriate distance metric to use?


- What share of the evaluated grid positions should be used to pre-filter sampled grid positions?
    
    What share of the already evaluated grid positions should be used to base pre-filtering on? Currently the top 20% of evaluated grid positions are used, unless that value is below 3 or over 20, in which case 3 or 20 are used, respectively.