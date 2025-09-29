# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.
## POLICY ITERATION ALGORITHM
-> Step1 : We are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2: Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.

## POLICY IMPROVEMENT FUNCTION
### Name : JEEVANESH S
### Register Number : 212222243002
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
### Name : JEEVANESH S
### Register Number : 212222243002
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```
## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="1163" height="272" alt="image" src="https://github.com/user-attachments/assets/490a6b5c-3a55-48e8-87d4-f4235dff2b5c" />


### 2. Policy, Value function and success rate for the Improved Policy
<img width="842" height="156" alt="image" src="https://github.com/user-attachments/assets/2a615043-e4d3-4644-8774-97e7bedb62f8" />
<img width="980" height="122" alt="image" src="https://github.com/user-attachments/assets/00ca1a37-04d7-4f21-bec6-c1959405fdf1" />
### 3. Policy, Value function and success rate after policy iteration
<img width="1105" height="170" alt="image" src="https://github.com/user-attachments/assets/28907f67-0761-441e-8d64-06d0a4b97be3" />
<img width="1010" height="47" alt="image" src="https://github.com/user-attachments/assets/e3ad9738-66e6-4941-abd2-d9fb55e58b5e" />
## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
