Initialize agent parameters:
    - Number of time bins: \( N_{\text{time\_bins}} \)
    - Hidden layer size: \( H \)
    - Learning rate: \( \alpha \)
    - Discount factor: \( \gamma \)

**Discretize Action Space:**
- For each location \( L \in \{0, 1, ..., m\} \):
  - Determine \( T_{\text{data\_lower}}[L], T_{\text{data\_upper}}[L] \)
  - If \( T_{\text{data\_lower}} = T_{\text{data\_upper}} \):
    - \( \text{TimeValues} \leftarrow \{ T_{\text{data\_lower}} \} \)
  - Else:
    - \( \text{TimeValues} \leftarrow \) equally spaced values between \( T_{\text{data\_lower}}[L] \) and \( T_{\text{data\_upper}}[L] \) with \( N_{\text{time\_bins}} \) points
  - For each \( T_{\text{data}} \in \text{TimeValues} \):
    - Add action \( (L, T_{\text{data}}) \) to action list

**Initialize Neural Networks:**
- Policy network \( Q(s, a; \theta) \)
- Target network \( Q'(s, a; \theta^{-}) \) with weights \( \theta^{-} \leftarrow \theta \)

**Initialize Replay Memory:** \( D \)

**For** each episode **do:**
1. **Initialize state** \( s_0 \)

2. **For** each time step \( t \) **do:**
   - **Get valid actions** \( A_{\text{valid}}(s_t) \) using function `GetValidActions(s_t)`
   - **Select action** \( a_t \) using ε-greedy policy:
     - With probability \( \epsilon \), select random \( a_t \in A_{\text{valid}}(s_t) \)
     - Else, \( a_t \leftarrow \arg\max_{a \in A_{\text{valid}}(s_t)} Q(s_t, a; \theta) \)
   - **Execute action** \( a_t \), observe reward \( r_t \) and next state \( s_{t+1} \)
   - **Store transition** \( (s_t, a_t, r_t, s_{t+1}, \text{done}) \) in \( D \)
   - **Sample minibatch** of transitions \( (s_j, a_j, r_j, s_{j+1}, \text{done}_j) \) from \( D \)
   - **Compute target** for each transition:
     - **If** \( \text{done}_j \):
       - \( y_j \leftarrow r_j \)
     - **Else:**
       - \( y_j \leftarrow r_j + \gamma \max_{a' \in A_{\text{valid}}(s_{j+1})} Q'(s_{j+1}, a'; \theta^{-}) \)
   - **Update policy network** by minimizing loss:
     - \( L(\theta) = \frac{1}{|\text{minibatch}|} \sum_j \left( y_j - Q(s_j, a_j; \theta) \right)^2 \)
   - **Every** \( C \) steps, update target network:
     - \( \theta^{-} \leftarrow \theta \)
   - **If** \( \text{done} \):
     - **Break** loop

**End For**

**Function GetValidActions(\( s \)):**
- Initialize valid actions set \( A_{\text{valid}}(s) \leftarrow \emptyset \)
- Extract state variables from \( s \):
  - Current location \( L_t \)
  - Remaining time \( T_{t}^{\text{rem}} \)
  - Visited locations \( V_t \)
  - Weather \( W_t \)
- **For** each action \( (L_{\text{next}}, T_{\text{data\_next}}) \) in action list **do:**
  - **If** \( L_{\text{next}} \neq 0 \) and \( V_t[L_{\text{next}}] = 1 \):
    - **Continue** (invalid action)
  - **If** \( T_{\text{data\_next}} \notin [T_{\text{data\_lower}}[L_{\text{next}}], T_{\text{data\_upper}}[L_{\text{next}}]] \):
    - **Continue** (invalid time)
  - Compute \( T_{\text{flight\_to\_next}} \leftarrow \text{GetFlightTime}(L_t, L_{\text{next}}, W_t) \)
  - Compute expected return time \( T_{\text{return}} \leftarrow \text{ExpectedReturnTime}(L_{\text{next}}) \)
  - **If** \( T_{t}^{\text{rem}} < T_{\text{flight\_to\_next}} + T_{\text{data\_next}} + T_{\text{return}} \):
    - **Continue** (insufficient time)
  - **Add** \( (L_{\text{next}}, T_{\text{data\_next}}) \) to \( A_{\text{valid}}(s) \)
- **Return** \( A_{\text{valid}}(s) \)

**Function GetFlightTime(\( L_{\text{from}}, L_{\text{to}}, W \)):**
- **If** \( W = 0 \):
  - **Return** \( T_{\text{flight\_good}}[L_{\text{from}}][L_{\text{to}}] \)
- **Else:**
  - **Return** \( T_{\text{flight\_bad}}[L_{\text{from}}][L_{\text{to}}] \)

**Function ExpectedReturnTime(\( L \)):**
- \( T_{\text{return\_good}} \leftarrow T_{\text{flight\_good}}[L][0] \)
- \( T_{\text{return\_bad}} \leftarrow T_{\text{flight\_bad}}[L][0] \)
- **Return** \( p \times T_{\text{return\_good}} + (1 - p) \times T_{\text{return\_bad}} \)
