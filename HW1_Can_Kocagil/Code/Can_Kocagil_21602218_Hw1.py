# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# As a basic numerical computation :
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import sys

question = input('Please Enter Question Number: (1/2)')


def Can_Kocagil_21602218_Hw1(question):
    if question == '1' :
        # %%
        print('QUESTION 1 part (a) \n')

        # Let's create the matrix A :
        A = np.array([[1, 0, -1, 2],
                    [2, 1, -1, 5],
                    [3, 3, 0, 9]])


        # Since alpha ve beta are arbitrary scalars:
        alpha, beta = (np.random.randn() , np.random.randn())


        # Hand driven solution to the system of Ax = b, x_n is:
        x_n = np.array([[alpha - 2 * beta],
                    [-alpha - beta],
                    [alpha],
                    [beta]])


        # Verification of the solution x_n to the system A * x_n = 0 :
        print(f"Proof that x_n solves the linear system of Ax = b is \n {A @ x_n} \n")


        Q1_TEST = lambda x_n : np.isclose(np.zeros((3,1)), A @ x_n)


        print(f'Q1 Verification \n {Q1_TEST(x_n)}')


        # %%
        print('QUESTION 1 part (b) \n')

        # Let's create the matrix A :
        A = np.array([[1, 0, -1, 2],
                    [2, 1, -1, 5],
                    [3, 3, 0, 9]])


        # Let's create output vector b :
        b = np.array([[1],
                    [4],
                    [9]])


        # Particular solution to the system A * x_p = b :
        x_p = np.array([[1],
                    [2],
                    [0],
                    [0]])


        # Verification of the solution x_n to the system A * x_p = 0 :
        print(f"Proof that x_p solves the linear system of Ax = b is \n {A @ x_p} \n")


        Q1_TEST = lambda x_p : np.isclose(b, A @ x_p)


        print(f'Q1 Verification \n {Q1_TEST(x_p)}')


        # %%
        print('QUESTION 1 part (c) \n')


        # Let's create the matrix A :
        A = np.array([[1, 0, -1, 2],
                    [2, 1, -1, 5],
                    [3, 3, 0, 9]])


        # Since alpha ve beta are arbitrary scalars:
        alpha, beta = (np.random.randn() , np.random.randn())


        # Hand driven solution to the system of Ax = b, x_general is:
        x_general = np.array([[alpha - 2 * beta + 1],
                            [-alpha - beta + 2],
                            [alpha],
                            [beta]])



        # Verification of the solution x_general to the system A * x_general = b :
        print(f"Proof that x_general solves the linear system of Ax = b is \n {A @ x_general} \n")


        Q1_TEST = lambda x_general : np.isclose(b, A @ x_general)


        print(f'Q1 Verification \n {Q1_TEST(x_general)}')


        # %%
        print('QUESTION 1 part (d) \n')

        # Let's apply SVD on matrix A :
        U, S, V_T = np.linalg.svd(A)


        # Little bit of calculation :
        (m,n) = A.shape
        S_plus = np.zeros((m,n))
        S_plus[:m, :m] = np.diag(np.concatenate((1 / S[0:2], np.array([0]))))

        print(f"Pseudo-inverse of A, A_plus is \n {V_T.T @ S_plus.T @ U.T} ")



        Q1_TEST = lambda S_plus : np.isclose( np.linalg.pinv(A), V_T.T @ S_plus.T @ U.T)


        print(f'Q1 Verification \n {Q1_TEST(S_plus)}')


        # %%
        print('QUESTION 1 part (e) \n')
        #from tabulate import tabulate
        # Our hand-driven alpha and beta values :
        alphas = [1,0,0,0,-1,2]
        betas  = [1,0,.5,2,0,0]

        # Let's see whether our alpha-beta values are correct or not:
        table = [[(s_alpha,s_beta),np.array([[s_alpha - 2 * s_beta + 1],
                                            [-s_alpha - s_beta + 2],
                                            [s_alpha],
                                            [s_beta]]).T,(A @ np.array([[s_alpha - 2 * s_beta + 1],
                                            [-s_alpha - s_beta + 2],
                                            [s_alpha],
                                            [s_beta]])).T] for  s_alpha,s_beta in zip(alphas,betas)]

        #print(tabulate(table,headers = ['Alpha-Beta','Sparsest x',' A dot Sparsest X'],tablefmt = 'fancy_grid'))


        # %%
        print('QUESTION 1 part (f) \n')

        print(f"The least norm solution to the system is \n {np.linalg.pinv(A) @ b}")

                ##question 1 code goes here

    elif question == '2' :
        # %%
        print('QUESTION 2 part (a) \n')


        # Given probability ranges :
        prob_range = np.arange(0, 1.00, 0.001)
        language = [binom.pmf(k = 103, n = 869 , p = prob) for prob in prob_range]
        not_language = [binom.pmf(k = 199, n = 2353, p = prob) for prob in prob_range]

        def plot_likelihood(likelihood : list or np.ndarray,
                            xticks : tuple or np.ndarray = (0, 0.05, 0.1, 0.15, 0.2),
                            xtick1 : np.ndarray = np.arange(0, 201, step=50),                    
                            color  : str = 'orange',
                            xlim         = None,                     
                            xlabel : str = 'Probability Range',
                            ylabel : str = 'Likehoods',
                            title  : str = 'Likelihood function of tasks involving language') -> None:

            """
            Given the likelihood array or list of float, plots the likelihood function w.r.t. 
            given probability range.

                Parameters:
                    - likelihood (list[float] or np.ndarray) : Likelihood function to be plotted
                    - xticks (tuple[float] or np.ndarray)    : set the current tick locations and labels of the x-axis
                    - color (str)                            : Color of the figure
                    - xlim (int)                             : The limit of the x label
                    - xlabel (str)                           : The text of x label
                    - ylabel (str)                           : The text of y label
                    - title (str) :                          : The title of the figure

                Returns:
                    - None


            """
            
            plt.figure(figsize = (6,6))
            plt.bar(np.arange(len(likelihood)), likelihood, color = color)  

            if xlim is not None:  
                plt.xlim(0, xlim)
            
            plt.xticks(xtick1, xticks )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show(block=False)


        plot_likelihood(language,
                        xlim = 200)
        plot_likelihood(not_language,
                        color = 'green',
                        title = 'Likelihood function of tasks not involving language',
                        xlim = 200)


        # %%
        print('QUESTION 2 part (b) \n')

        table = [ [task, prob_range[np.argmax(likelihood)], max(likelihood)] for likelihood,task in zip([language,not_language], ['Language Involving Tasks', 'Language Not Involving Tasks'])]

        #print(tabulate(table,headers = ['Tasks','Probability that maximixes','Maximum value'],tablefmt = 'fancy_grid'))
        print(table)


        # %%
        print('QUESTION 2 part (c) \n')

        def bayes_theorem(likelihood : np.ndarray, prior : float) -> np.ndarray:
            """
            Given the likelihood function and prior distribution,
            computes and returns the posterior distribution by Bayes' Rule

                Parameters:
                    - likelihood (np.ndarray) : likelihood function (e.g., language or not_language)
                    - prior (float)           : prior distribution as a probability value 

                Returns:
                    - Posterior probability (np.ndarray) with normalization


            """

            # Normalizing all likelihood values
            normalization_constant = np.sum(likelihood * prior)
            
            # Computing posterior distribution    
            posterior = likelihood * prior 

            return posterior/normalization_constant


        uniform_prior = 1 / len(prob_range)

        posterior_language = bayes_theorem(np.array(language),uniform_prior)
        plot_likelihood(likelihood = posterior_language,
                        color = 'b',
                        title = 'Posterior distribution for language involving tasks',
                        xlim = 200)


        posterior_not_language = bayes_theorem(np.array(not_language),uniform_prior)
        plot_likelihood(likelihood = posterior_not_language,
                        color = 'purple',
                        title = 'Posterior distribution for not language involving tasks',
                        xlim = 200)


        # %%
        # Calculating CDF of language involving tasks and plottings:
        posterior_language_cdf = [np.sum(posterior_language[:until]) for until in range(1, len(prob_range) + 1)]

        plot_likelihood(likelihood = posterior_language_cdf,
                        xticks     = np.around(np.arange(0, 1.001, 0.1), 2),
                        xtick1     = np.arange(0, 1001, 100),
                        title      = 'Cumulative distribution of tasks involving language',
                        ylabel     = 'Cumulative Distribution Function (CDF)',
                        color      = 'm')


        # Calculating CDF of not language involving tasks and plottings:
        posterior_not_language_cdf = [np.sum(posterior_not_language[:until]) for until in range(1, len(prob_range) + 1)]

        plot_likelihood(likelihood = posterior_not_language_cdf,
                        xticks     = np.around(np.arange(0, 1.001, 0.1), 2),
                        xtick1     = np.arange(0, 1001, 100),
                        title      = 'Cumulative distribution of tasks not involving language',
                        ylabel     = 'Cumulative Distribution Function (CDF)',
                        color      = 'red')


        # %%
        # Efficients calculations of CDF:
        cdf_language = np.empty(len(posterior_language))
        for i in range(len(posterior_language)):
            if i == 0:
                cdf_language[i] = posterior_language[i]
            else:
                cdf_language[i] = cdf_language[i-1] + posterior_language[i]


        plot_likelihood(likelihood = cdf_language,
                        xticks     = np.around(np.arange(0, 1.001, 0.1), 2),
                        xtick1     = np.arange(0, 1001, 100),
                        title      = 'Cumulative distribution of tasks involving language',
                        ylabel     = 'Cumulative Distribution Function (CDF)',
                        color      = 'm')

        # Efficients calculations of CDF:
        posterior_not_language_cdf = np.empty(len(posterior_not_language))
        for i in range(1, len(posterior_not_language)):
            if i == 0:
                posterior_not_language_cdf[i] = posterior_not_language[i]
            else:
                posterior_not_language_cdf[i] = posterior_not_language_cdf[i-1] + posterior_not_language[i]


        plot_likelihood(likelihood = posterior_not_language_cdf,
                        xticks     = np.around(np.arange(0, 1.001, 0.1), 2),
                        xtick1     = np.arange(0, 1001, 100),
                        title      = 'Cumulative distribution of tasks not involving language',
                        ylabel     = 'Cumulative Distribution Function (CDF)',
                        color      = 'g')


        # %%
        lower_bound = 0.025
        upper_bound = 0.975
        flags = [True] * 4

        i = 0
        while any(flags) and i < len(prob_range):

            
            if cdf_language[i] >= lower_bound and flags[0]:
                lower_confidence_interval_l = prob_range[i]
                flags[0] = False
                

            if cdf_language[i] >= upper_bound and flags[1]:
                higher_confidence_interval_l = prob_range[i]
                flags[1] = False

            if posterior_not_language_cdf[i] >= lower_bound and flags[2]:
                lower_confidence_interval_nl = prob_range[i]
                flags[2] = False
                

            if posterior_not_language_cdf[i] >= upper_bound and flags[3]:       
                higher_confidence_interval_nl = prob_range[i]
                flags[3] = False


            i += 1
                
                
        print(f"Lower 95% confidence for language involving tasks likelihood CDF {lower_confidence_interval_l} ")

        print(f"Higher 95% confidence for language involving tasks likelihood CDF {higher_confidence_interval_l} ")

        print(f"Lower 95% confidence for language not involving tasks likelihood CDF {lower_confidence_interval_nl} ")

        print(f"Higher 95% confidence for language not involving tasks likelihood CDF {higher_confidence_interval_nl} ")


        # %%
        print('QUESTION 2 part (d) \n')
        plt.figure()
        joint = np.outer(posterior_language.T,posterior_not_language)
        plt.imshow(joint)
        plt.colorbar()
        plt.title('The joint posterior distribution')
        plt.xlabel('Language Involving RV (X_l)')
        plt.ylabel('Language Not Involving RV (X_nl)')
        plt.xticks(np.arange(len(posterior_language), step=100),
                np.round(np.arange(0.1,1.1,0.1),3))
        plt.yticks(np.arange(len(posterior_language), step=100),
                np.round(np.arange(0.1,1.1,0.1),3))
        plt.show(block=False)


        # %%
        # computing the P(X_l > X_nl | data) and P(X_nl >= X_l | data)

        assert len(posterior_language) == len(posterior_not_language)

        lower_tri_sum = 0
        upper_and_diag_tri_sum = 0
        for i in range(len(posterior_language)):
            for j in range(len(posterior_not_language)):
                if i > j:
                    lower_tri_sum += joint[i,j]
                else:
                    upper_and_diag_tri_sum += joint[i,j]


        print(f"Sum of entries of lower triangle of joint distribution (i.e., P(X_l > X_nl | data)) = {lower_tri_sum} \n")

        print(f"Sum of entries of upper triangle and diagonal of joint distribution (i.e., P(X_nl >= X_l | data) )  = {upper_and_diag_tri_sum}")    


        # %%
        print('QUESTION 2 part (e) \n')
        # Here, let's recall and recompute the conditional probabilities :
        max_prop_l  = prob_range[np.argmax(language)]
        max_prop_nl = prob_range[np.argmax(not_language)]

        # P(Language) = 0.5 is given :
        p_language = .5

        # Here is the Bayes' Rule for inferensing: 
        p_language_given_activation = max_prop_l * p_language / ( ( max_prop_l * p_language) + (max_prop_nl * (1-p_language)))


        print(f"P(language|activation) = {p_language_given_activation} ")        
        
    else:
        print('Wrong Number question, please give 1 or 2')



Can_Kocagil_21602218_Hw1(question)
