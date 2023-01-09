library(tidyverse)
library(fdrtool)

filename = "RTFM"

data = read_csv(paste0("./Output/", filename, ".csv"))

# First off, we need to calculate z-values from the obtained epsilon-values.
mu = mean(data$epsilon)
sigma = sd(data$epsilon)

data = data %>%
  mutate(z = (epsilon - mu) / sigma )

# Given these z-values, we can estimate f(z), define the null density from the data, and calculate the false discovery rate.
# To that end, we can use the R package fdrtool as recommended by Kleinberg.

# fdr = fdrtool(data$z) # Did not work for the RTFM search space
fdr = fdrtool(data$z, cutoff.method = "pct0", pct0 = 0.95)

# Add the fdr values to the dataset, and see which causes are significant for the effect.
data$fdr = fdr[["lfdr"]]

# Set a threshold for the FDR to determine significance.
threshold = 0.05
data %>% filter(fdr <= threshold)

write_csv(data, paste0("./output/", filename, "_causal_sig.csv"))