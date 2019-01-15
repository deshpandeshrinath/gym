# Environment for Path Synthesis

The task is to synthesize a Four-R linkage for given target path with 
normalized cross-correlation score of 0.98 or above.

# State

- Uniformly Sampled Moving Joint coordinates throughout the entire simulation in
one branch.
  - shape of [7, 20], where 7 corresponds to xm_1, ym_1, xm_2, ym_2, xc, yc, \theta_c.
- Linkage ratios
  - i.e. (l_1, l_2, l_3, l_4, l_5)

Thus total of 7*20 + 5 = 145 dimensional vector
