# Quickest Change Detection for Unnormalized Statistical Models
This is an implementation of [Quickest Change Detection for Unnormalized Statistical Models](https://arxiv.org/abs/2302.00250)

## Requirements
See `requirements.txt`

## Instructions
 - Global hyperparameters are configured in `config.yml`
 - Use `make.sh` to generate run script
 - Use `make.py` to generate exp script
 - Use `process.py` to process exp results
 - Experimental setup are listed in `make.py` 
 - Hyperparameters can be found at `process_control()` in utils.py 
 - `modules/cpd.py` defines Change Point Detection methods
 
## Examples
 - Test CUSUM for MVN ( $d=2$ ) dataset with 500 pre data, 10000 post data, $\epsilon_{\mu} = 0.1$, no noise, ARL $=2000$
    ```ruby
    python test_cpd.py MVN-2_500_10000_0.1-0.0_0_2000
    ```
 - Test Scan B-statistic MVN EXP ( $d=2$ ) dataset with 500 pre data, 10000 post data, $\epsilon_{\log \sigma^2} = 0.5$, $\sigma_{noise} = 0.1$, ARL $=2000$
    ```ruby
    python test_cpd.py MVN-2_500_10000_0.0-0.5_0.1_2000
    ```
 - Test CALM-MMD for EXP ( $d=2$ ) dataset with 500 pre data, 10000 post data, $\epsilon_{\tau} = 1.0$, $\sigma_{noise} = 0.3$, ARL $=2000$
    ```ruby
    python test_cpd.py EXP-2_500_10000_1.0_0.3_2000
    ```
 - Test SCUSUM for RBM ( $d=50$ ) dataset with 500 pre data, 10000 post data, $\epsilon_{\log \sigma^2} = 0.05$, no noise, ARL $=2000$, $m=500$
    ```ruby
    python test_cpd.py RBM-50_500_10000_0.05_0_2000_500
    ```

## Results
- The results of Detection Score (before and after change) with MVN ( $\epsilon_{\mu} = 0.3$ ) and ARL $=2000$.
<p align="center">
<img src="/asset/MVN-2_500_10000_0.0-0.3_0_scusum_2000_score_mean.png">
</p>

## Acknowledgements
*Suya Wu  
Enmao Diao  
Taposh Banerjee  
Jie Ding  
Vahid Tarokh*
