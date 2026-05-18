# Turbofan RUL Prediction and Maintenance Policy

**TL;DR** Sensor data from 100 simulated jet engines, predict how many cycles until failure. LSTM halves GBM's error and cuts dangerous late predictions by four orders of magnitude. DQN maintenance policy trains but loses to a fixed threshold, which is itself a finding.

Predicting when industrial equipment will fail is a core reliability engineering problem with direct consequences for maintenance cost, unplanned downtime, and safety. This project applies degradation modeling and reinforcement learning to the NASA CMAPSS turbofan dataset, a physics-based simulation of jet engine wear across hundreds of operating cycles. The methods transfer to any domain where sensors track a system's slow decline toward failure, including rotating machinery, power electronics, and civil infrastructure, making the approach more broadly applicable than the aerospace framing suggests.

## Dataset

The CMAPSS FD001 subset simulates 100 turbofan engines under a single operating condition with one active fault mode. Each row is one engine-cycle observation with 26 columns: engine ID, cycle number, three operational settings, and 21 sensor readings covering temperature, pressure, fan speed, and fuel flow. Engines run from a healthy baseline to failure, with lifetimes ranging from roughly 130 to 360 cycles. The prediction target (remaining useful life, or RUL) is derived by counting cycles backward from each engine's last recorded observation.

## Methods

Notebooks 01 and 02 cover exploratory analysis and feature engineering: constant-variance sensors and single-condition operational settings are dropped, and surviving sensors are scaled per engine to isolate degradation trajectory rather than absolute sensor level. Notebook 03 trains a LightGBM regressor on 5-cycle rolling statistics as an interpretable baseline. Notebook 04 trains a two-layer LSTM on 30-cycle sliding windows, with a capped RUL target of 125 cycles to prevent mode collapse, and notebook 05 compares both models head to head. Notebook 06 frames maintenance scheduling as a reinforcement learning problem, training a DQN agent to decide when to replace an engine given its predicted RUL.

## Results

| Model | RMSE | NASA Score |
|---|---:|---:|
| GBM baseline | 47.15 | 38,175,097 |
| LSTM | 20.83 | 19,490 |

The NASA scoring function applies an exponential penalty to late predictions, where predicted RUL exceeds actual RUL. The LSTM's four-order-of-magnitude improvement on that metric reflects fewer dangerous overestimates near end-of-life, not just lower average error.

## Reinforcement Learning

The DQN agent observes predicted RUL at each cycle and chooses between holding and triggering a replacement. Rewards are structured to encode operational cost: +1 per safe cycle, +20 for replacement within 30 cycles of failure, -10 for premature replacement, and -100 for allowing an engine to fail without intervention.

FD001's degradation is predictable enough that a fixed threshold beats a learned policy. The DQN converged to never replacing because grinding +1 per cycle and eating the occasional -100 failure still averages close to what the rule-based policy earns. The replacement window is too narrow for random exploration to reliably discover. This isn't a dead end. On FD003 and FD004 where engines operate across multiple conditions, fixed thresholds stop working and a learned policy has real room to win. That extension is the logical next step.

## Stack

Python, pandas, numpy, LightGBM, PyTorch, stable-baselines3, gymnasium, Google Colab T4.

## What's Next

- Extend to FD002-FD004 multi-condition datasets where operating regime adds signal the single-condition models never see.
- Try a Transformer-based sequence model as an LSTM alternative.
- Retrain the DQN on multi-condition data where a learned policy has a genuine edge over fixed thresholds.
