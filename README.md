# Codes for Master's Thesis AY25/26

This project proposes a method for calibrating a Heston Local Stochastic Volatility model by applying Gyongy’s mimicking theorem on a Schrodinger Local Volatility model. Within this project, the market risk-neutral densities are recovered from market OTM options, then a piecewise time-homogeneous Local Volatility function is calibrated to bridge densities between fixed maturity dates. A Continuous Time Markov Chain Heston Local Stochastic Volatility model with ρ=0 is first calibrated to match the marginals of the Local Volatility model, and then it is extended to a full Continuous Time Markov Chain Heston Local Stochastic Volatility model with ρ̸=0, which is calibrated using the Lamperti transform. Then price the vanilla and autocallable options using all calibrated models through the matrix exponential of infinitesimal generators.

Option data and zero-coupon rates on 2 Jan, 2025 were used.

CTMC_LSV_Calibration and CTMC_Lamperti_LSV_Calibration codes need to be rerun to obtain .npz result files for autocallable pricing. Place the result files in the data folder in the Autocallable folder.
