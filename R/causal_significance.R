library(tidyverse)
library(stringr)
library(qvalue)
library(readxl)
library(xlsx)

wd <- "./Output/Finalised/"
file <- "VSI_Revision_VaryingTraces.xlsx"
sheets <- c("10.000 traces", "7500 traces", "5000 traces", "2500 traces", "1000 traces", "500 traces", "250 traces", "100 traces")

sheet <- "10.000 traces"

compute_q <- function(df) {
  mu = mean(df$epsilon)
  sigma = sd(df$epsilon)

  df = df %>%
    mutate(z = (epsilon - mu) / sigma ) %>%
    mutate(p = pnorm(z, lower.tail = F))

  q <- qvalue_truncp(df$p)
  q_df <- data.frame(p = q$pvalues, q = q$qvalues)
  df <- left_join(df, q_df)
  return(df)
}

for (sheet in sheets) {
  # read the file
  df <- read_excel(paste0(wd, file), sheet = sheet)

  # perform computations
  df <- compute_q(df)

  # Write the results to the file
  xlsx::write.xlsx(df, paste0(wd, str_replace(file, '.xlsx', '_q.xlsx')), sheetName = sheet, append = TRUE)
}

# Genuine causes filtered out
df <- read_excel(paste0(wd, "VSI_Revision_NoGenuines.xlsx"))
df <- compute_q(df)
xlsx::write.xlsx(df, paste0(wd, "VSI_Revision_NoGenuines_q.xlsx"))

# RTFM - first case study
df <- read_excel(paste0(wd, "RTFM.xlsx"))
df <- compute_q(df)
xlsx::write.xlsx(df, paste0(wd, "RTFM_q.xlsx"))
