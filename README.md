DQN on CartPole-v1: Baseline, Reward Shaping, and Hyperparameter Study
Name: Gowtham Vuppaladhadiam 
Environment: CartPole-v1 (Gymnasium)

1)	Abstract
We trained a DQN-style agent with a fixed hyperparameter set on three control tasks: env, angle, and angle_pos. Success was defined as a rolling average return over the last 100 episodes (Avg100) ≥ 195.0. The agent solved env at episode 398 and angle at episode 319, repeatedly achieving single-episode returns of 500 thereafter. On angle_pos, performance plateaued around 8–35 with intermittent spikes up to ~123 and never met the solve criterion. We attribute the failure primarily to observation difficulty and value-estimation instability. We propose concrete remedies (Double+Dueling DQN, prioritized replay, n-step targets, adjusted target-network update cadence, and reward/feature adjustments) to close the gap.

2)	Problem Statement & Success Criterion
•	Objective: Learn policies that maximize episodic return for each task.
•	Solve Criterion: Avg100 ≥ 195.0.
•	Tasks: env, angle, angle_pos.

2) Method (DQN Skeleton)
•	Network: MLP [4, 128, 128, 2] with ReLU.
•	Replay buffer: 10,000 transitions; sample batch=64.
•	Optimizer: Adam, LR=1e-3.
•	Discount: γ = 0.99.
•	Target network: hard update every 100 steps.
•	Exploration: ε-greedy 1.0 → 0.01, ×0.995 per episode (clamped at 0.01).
•	Seed: 42.
•	Implementation detail: We store either the env reward (baseline) or a shaped reward in replay, but success is judged by the env reward.
3)	Experimental Setup
•	Shared Hyperparameters (all tasks):
o	Learning rate (LR): 0.001
o	Target-network update: every 100 steps
o	Batch size: 64
o	ε-decay factor: 0.995 (ε started ≈0.995; reached ~0.20 by solved runs)
•	Training protocol:
o	Single-seed runs reported here.
o	Replay buffer warm-up before learning (standard DQN practice).
o	Performance tracked per episode and via Avg100.

4)	Evaluation Protocol
•	Primary metric: Avg100 (rolling mean of the last 100 episode returns).
•	Secondary metrics: Best single-episode return, qualitative stability after first solve.

5)	Results
Task	Solved?	Episode Solved	Best Single-Ep Return	End Behavior (qualitative)
env	✅	398	500 (repeated)	Frequent 200–500 returns post-solve; stable
angle	✅	319	500 (repeated)	Strong late-stage stability; many 200–500
angle_pos	❌	–	~123	Avg100 never near 195; long low-return plateau with sporadic 70–120 spikes

6)	 Learning Dynamics (highlights)
•	env: Early Avg100 ~18→34 (eps 1–100) with first notable spikes by ep 78 (≈109). Steady mid-phase gains; crossed Avg100=195 at ep 398. Many 300–500 episodes afterward.
•	angle: Slightly faster ramp. First ≥100 at ep 62 (≈118). Multiple 200–500 episodes from ~ep 173 onward, solving by ep 319.
•	angle_pos: Prolonged 8–35 plateau. Occasional spikes (70–120, peak ~123, ~ep 711), but no sustained high-return regime; solve criterion unmet.
 
7)	Hyperparameter experiment
Hyperparameter Sweep Results (env reward)    
baseline 	 solved_ep=398 	 lr=0.001 	 tgtUpd=100 	 batch=64  	 epsDecay=0.995
faster_target 	 solved_ep=299 	 lr=0.001 	 tgtUpd=50  	 batch=64  	 epsDecay=0.995
slower_target 	 solved_ep=387 	 lr=0.001 	 tgtUpd=200 	 batch=64  	 epsDecay=0.995
    lower_lr 	 solved_ep=429 	 lr=0.0005 	 tgtUpd=100 	 batch=64  	 epsDecay=0.995
   higher_lr 	 solved_ep=363 	 lr=0.002 	 tgtUpd=100 	 batch=64  	 epsDecay=0.995
bigger_batch 	 solved_ep=339 	 lr=0.001 	 tgtUpd=100 	 batch=128 	 epsDecay=0.995

Faster target updates (every 50 steps) improved stability and solved earlier (286) → more frequent target refresh helps with stale targets in CartPole.
Slower target (200) hurt stability/learning speed (512).
Lower LR (5e-4) slowed learning moderately; higher LR (2e-3) degraded stability and delayed solving—consistent with Q-learning’s sensitivity to step size.
Bigger batch (128) was slightly faster than baseline here (333 vs. 375), likely due to lower gradient variance.

8)	Reproducibility
•	Code: instructor DQN skeleton with minimal additions (reward shaping function; per-episode ε decay; solved/plots; small sweep loop).
•	Libs: PyTorch, Gymnasium; device torch.device("cuda" if available else "cpu").
•	Seed: 42 for Python/NumPy/Torch + env/action space.
•	Run budget: up to 800 episodes per run (stops early when solved).
•	Differences from skeleton: only reward shaping line (the reward stored to memory) and book-keeping for moving averages/plots.

File URL: DQN/DQN_execution_Gowtham.ipynb at main · gowthamvsn/DQN
