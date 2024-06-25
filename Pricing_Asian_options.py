#!/usr/bin/env python
# coding: utf-8

# # Project A : Malliavin

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x=100
r=0.03
sigma=0.2
T=1

K1=100
K2=110


# In[3]:


def trajectories_and_gaussian(x, r, sigma, T, M, N, seed = 42):
    
    np.random.seed(seed)
    
    gaussian=np.random.randn(N, M+1)
    exp_gaussian=np.exp((r-(1/2)*(sigma**2))*(T/M)+sigma*np.sqrt(T/M)*gaussian)
    exp_gaussian[:, 0] = x
    trajectories=np.cumprod(exp_gaussian,axis=1)
    
    return trajectories, gaussian


# In[4]:


def monte_carlo(sample):
    
    estimate_expectation = np.mean(sample)
    
    #calculation of the 95% confidence interval
    std_dev = np.std(sample)
    confidence_interval=estimate_expectation-1.96 * std_dev/np.sqrt(len(sample)), estimate_expectation+1.96 * std_dev/np.sqrt(len(sample))
    
    return [estimate_expectation,std_dev**2,confidence_interval]


# In[5]:


def pricing(N, M):
    # List of N
    X = list(range(1000, N+1, 1000))
    # Lists for prices, P1 for option 1 and P2 for option 2
    P1 = []
    P2 = []
    # Lists for empirical variances of the estimator
    V1 = []
    V2 = []
    for i in X:
        trajectories = trajectories_and_gaussian(x, r, sigma, T, M, i, seed=42)[0]
        stock_integral = (T/M) * np.sum(trajectories, axis=1)
        payoff1 = np.exp(-r*T) * np.maximum(stock_integral-K1, 0)
        payoff2 = np.exp(-r*T) * np.where((stock_integral >= K1) & (stock_integral <= K2), 1, 0)
        P1.append(monte_carlo(payoff1)[0])
        P2.append(monte_carlo(payoff2)[0])
        V1.append(monte_carlo(payoff1)[1])
        V2.append(monte_carlo(payoff2)[1])
        
        if i == N:
            result1 = monte_carlo(payoff1)
            result2 = monte_carlo(payoff2)
            print(f"For N={N} and M={M}:\n"
                  f"Option 1:\n"
                  f"\t- Estimated price: {result1[0]}\n"
                  f"\t- Empirical variance: {result1[1]}\n"
                  f"\t- 95% confidence interval: {result1[2]}\n"
                  f"Option 2:\n"
                  f"\t- Estimated price: {result2[0]}\n"
                  f"\t- Empirical variance: {result2[1]}\n"
                  f"\t- 95% confidence interval: {result2[2]}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot for P1 against X on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('N (Number of simulations)')
    ax1.set_ylabel('Estimated Price', color=color)
    ax1.plot(X, P1, color=color, label='Estimated Price')
    ax1.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V1
    ax1_2 = ax1.twinx()
    color = 'tab:red'
    ax1_2.set_ylabel('Empirical Variance', color=color)
    ax1_2.plot(X, V1, color=color, label='Empirical Variance')
    ax1_2.tick_params(axis='y', labelcolor=color)

    # Title for the first plot
    ax1.set_title(f'Option 1: Evolution of Estimated Price and Empirical Variance with N for M={M}')

    # Combine legends for the first plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1_2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Plot for P2 against X on the secondary y-axis
    color = 'tab:blue'
    ax2.set_xlabel('N (Number of simulations)')
    ax2.set_ylabel('Estimated Price', color=color)
    ax2.plot(X, P2, color=color, label='Estimated Price')
    ax2.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V2
    ax2_2 = ax2.twinx()
    color = 'tab:red'
    ax2_2.set_ylabel('Empirical Variance', color=color)
    ax2_2.plot(X, V2, color=color, label='Empirical Variance')
    ax2_2.tick_params(axis='y', labelcolor=color)

    # Title for the second plot
    ax2.set_title(f'Option 2: Evolution of Estimated Price and Empirical Variance with N for M={M}')

    # Combine legends for the second plot
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2_2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()


# ### Question a)

# In[6]:


pricing(70000,50)
pricing(70000,150)
pricing(70000,250)


# ### Question b)

# In[7]:


def finite_difference(eps,N,M):
    # List of N
    X = list(range(1000, N+1, 1000))
    # Lists for deltas(x), D1 for option 1 and D2 for option 2
    D1 = []
    D2 = []
    # Lists for empirical variances of the estimator
    V1 = []
    V2 = []
    for i in X:
        
        gaussian=np.random.randn(N, M+1)
        
        exp_gaussian_plus=np.exp((r-(1/2)*(sigma**2))*(T/M)+sigma*np.sqrt(T/M)*gaussian)
        exp_gaussian_plus[:, 0] = x + eps
        
        exp_gaussian_minus=np.exp((r-(1/2)*(sigma**2))*(T/M)+sigma*np.sqrt(T/M)*gaussian)
        exp_gaussian_minus[:, 0] = x - eps
        
        trajectories_plus=np.cumprod(exp_gaussian_plus,axis=1)
        trajectories_minus=np.cumprod(exp_gaussian_minus,axis=1)
      
        stock_integral_plus = (T/M) * np.sum(trajectories_plus, axis=1)
        stock_integral_minus = (T/M) * np.sum(trajectories_minus, axis=1)
        
        payoff1_plus=np.exp(-r*T)*np.maximum(stock_integral_plus-K1,0)
        payoff1_minus=np.exp(-r*T)*np.maximum(stock_integral_minus-K1,0)

        payoff2_plus=np.exp(-r*T)*np.where((stock_integral_plus >= K1) & (stock_integral_plus <= K2), 1, 0)
        payoff2_minus=np.exp(-r*T)*np.where((stock_integral_minus >= K1) & (stock_integral_minus <= K2), 1, 0)

        derivative1=(payoff1_plus-payoff1_minus)/eps
        derivative2=(payoff2_plus-payoff2_minus)/eps

        D1.append(monte_carlo(derivative1)[0])
        D2.append(monte_carlo(derivative2)[0])
        V1.append(monte_carlo(derivative1)[1])
        V2.append(monte_carlo(derivative2)[1])
        
        if i == N:
            result1 = monte_carlo(derivative1)
            result2 = monte_carlo(derivative2)
            print(f"For N={N} and M={M}:\n"
                  f"Option 1:\n"
                  f"\t- Estimated delta: {result1[0]}\n"
                  f"\t- Empirical variance: {result1[1]}\n"
                  f"\t- 95% confidence interval: {result1[2]}\n"
                  f"Option 2:\n"
                  f"\t- Estimated delta: {result2[0]}\n"
                  f"\t- Empirical variance: {result2[1]}\n"
                  f"\t- 95% confidence interval: {result2[2]}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot for P1 against X on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('N (Number of simulations)')
    ax1.set_ylabel('Estimated Delta', color=color)
    ax1.plot(X, D1, color=color, label='Estimated Delta')
    ax1.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V1
    ax1_2 = ax1.twinx()
    color = 'tab:red'
    ax1_2.set_ylabel('Empirical Variance', color=color)
    ax1_2.plot(X, V1, color=color, label='Empirical Variance')
    ax1_2.tick_params(axis='y', labelcolor=color)

    # Title for the first plot
    ax1.set_title(f'Option 1 (FDM) : Evolution of Estimated Delta and Empirical Variance with N for M={M}')

    # Combine legends for the first plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1_2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Plot for P2 against X on the secondary y-axis
    color = 'tab:blue'
    ax2.set_xlabel('N (Number of simulations)')
    ax2.set_ylabel('Estimated Delta', color=color)
    ax2.plot(X, D2, color=color, label='Estimated Delta')
    ax2.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V2
    ax2_2 = ax2.twinx()
    color = 'tab:red'
    ax2_2.set_ylabel('Empirical Variance', color=color)
    ax2_2.plot(X, V2, color=color, label='Empirical Variance')
    ax2_2.tick_params(axis='y', labelcolor=color)

    # Title for the second plot
    ax2.set_title(f'Option 2 (FDM) : Evolution of Estimated Delta and Empirical Variance with N for M={M}')

    # Combine legends for the second plot
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2_2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()


# In[8]:


finite_difference(10**(-4),70000,50)
finite_difference(10**(-4),70000,150)
finite_difference(10**(-4),70000,250)

finite_difference(10**(-2),70000,50)
finite_difference(10**(-2),70000,150)
finite_difference(10**(-2),70000,250)


# ### Question c)

# In[9]:


def method_A(N,M):
    # List of N
    X = list(range(1000, N+1, 1000))
    # Lists for Deltas, D1 for option 1 and D2 for option 2
    D1 = []
    D2 = []
    # Lists for empirical variances of the estimator
    V1 = []
    V2 = []
    for i in X:
        trajectories, gaussian = trajectories_and_gaussian(x, r, sigma, T, M, i, seed=42)
        stock_integral = (T/M) * np.sum(trajectories, axis=1)
        
        Pi=gaussian[:, 1]/(x*np.sqrt(T/M)*sigma)
        
        function1=np.exp(-r*T)*np.maximum(stock_integral-K1,0)*Pi
        function2=np.exp(-r*T)*np.where((stock_integral >= K1) & (stock_integral <= K2), 1, 0)*Pi
        
        D1.append(monte_carlo(function1)[0])
        D2.append(monte_carlo(function2)[0])
        V1.append(monte_carlo(function1)[1])
        V2.append(monte_carlo(function2)[1])
        
        if i == N:
            result1 = monte_carlo(function1)
            result2 = monte_carlo(function2)
            print(f"For N={N} and M={M}:\n"
                  f"Option 1:\n"
                  f"\t- Estimated Delta: {result1[0]}\n"
                  f"\t- Empirical variance: {result1[1]}\n"
                  f"\t- 95% confidence interval: {result1[2]}\n"
                  f"Option 2:\n"
                  f"\t- Estimated Delta: {result2[0]}\n"
                  f"\t- Empirical variance: {result2[1]}\n"
                  f"\t- 95% confidence interval: {result2[2]}")
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot for P1 against X on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('N (Number of simulations)')
    ax1.set_ylabel('Estimated Delta', color=color)
    ax1.plot(X, D1, color=color, label='Estimated Delta')
    ax1.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V1
    ax1_2 = ax1.twinx()
    color = 'tab:red'
    ax1_2.set_ylabel('Empirical Variance', color=color)
    ax1_2.plot(X, V1, color=color, label='Empirical Variance')
    ax1_2.tick_params(axis='y', labelcolor=color)

    # Title for the first plot
    ax1.set_title(f'Option 1 (Method A) : Evolution of Estimated Delta and Empirical Variance with N for M={M}')

    # Combine legends for the first plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1_2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Plot for P2 against X on the secondary y-axis
    color = 'tab:blue'
    ax2.set_xlabel('N (Number of simulations)')
    ax2.set_ylabel('Estimated Delta', color=color)
    ax2.plot(X, D2, color=color, label='Estimated Delta')
    ax2.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V2
    ax2_2 = ax2.twinx()
    color = 'tab:red'
    ax2_2.set_ylabel('Empirical Variance', color=color)
    ax2_2.plot(X, V2, color=color, label='Empirical Variance')
    ax2_2.tick_params(axis='y', labelcolor=color)

    # Title for the second plot
    ax2.set_title(f'Option 2 (Method A) : Evolution of Estimated Delta and Empirical Variance with N for M={M}')

    # Combine legends for the second plot
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2_2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()    


# In[10]:


def method_B(N,M):
    # List of N
    X = list(range(1000, N+1, 1000))
    # Lists for Deltas, D1 for option 1 and D2 for option 2
    D1 = []
    D2 = []
    # Lists for empirical variances of the estimator
    V1 = []
    V2 = []
    for i in X:
        trajectories = trajectories_and_gaussian(x, r, sigma, T, M, i, seed=42)[0]
        stock_integral = (T/M) * np.sum(trajectories, axis=1)
        
        Pi=(2/x)*((trajectories[:, -1]-x-r*stock_integral)/((sigma**2)*stock_integral) + 1/2)

        function1=np.exp(-r*T)*np.maximum(stock_integral-K1,0)*Pi
        function2=np.exp(-r*T)*np.where((stock_integral >= K1) & (stock_integral <= K2), 1, 0)*Pi

        
        D1.append(monte_carlo(function1)[0])
        D2.append(monte_carlo(function2)[0])
        V1.append(monte_carlo(function1)[1])
        V2.append(monte_carlo(function2)[1])
        
        if i == N:
            result1 = monte_carlo(function1)
            result2 = monte_carlo(function2)
            print(f"For N={N} and M={M}:\n"
                  f"Option 1:\n"
                  f"\t- Estimated Delta: {result1[0]}\n"
                  f"\t- Empirical variance: {result1[1]}\n"
                  f"\t- 95% confidence interval: {result1[2]}\n"
                  f"Option 2:\n"
                  f"\t- Estimated Delta: {result2[0]}\n"
                  f"\t- Empirical variance: {result2[1]}\n"
                  f"\t- 95% confidence interval: {result2[2]}")
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot for P1 against X on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('N (Number of simulations)')
    ax1.set_ylabel('Estimated Delta', color=color)
    ax1.plot(X, D1, color=color, label='Estimated Delta')
    ax1.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V1
    ax1_2 = ax1.twinx()
    color = 'tab:red'
    ax1_2.set_ylabel('Empirical Variance', color=color)
    ax1_2.plot(X, V1, color=color, label='Empirical Variance')
    ax1_2.tick_params(axis='y', labelcolor=color)

    # Title for the first plot
    ax1.set_title(f'Option 1 (Method B) : Evolution of Estimated Delta and Empirical Variance with N for M={M}')

    # Combine legends for the first plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1_2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Plot for P2 against X on the secondary y-axis
    color = 'tab:blue'
    ax2.set_xlabel('N (Number of simulations)')
    ax2.set_ylabel('Estimated Delta', color=color)
    ax2.plot(X, D2, color=color, label='Estimated Delta')
    ax2.tick_params(axis='y', labelcolor=color)

    # Secondary y-axis for V2
    ax2_2 = ax2.twinx()
    color = 'tab:red'
    ax2_2.set_ylabel('Empirical Variance', color=color)
    ax2_2.plot(X, V2, color=color, label='Empirical Variance')
    ax2_2.tick_params(axis='y', labelcolor=color)

    # Title for the second plot
    ax2.set_title(f'Option 2 (Method B) : Evolution of Estimated Delta and Empirical Variance with N for M={M}')

    # Combine legends for the second plot
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2_2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()    


# In[11]:


method_A(70000,50)
method_A(70000,150)
method_A(70000,250)


# In[12]:


method_B(70000,50)
method_B(70000,150)
method_B(70000,250)


# In[ ]:





# In[ ]:




