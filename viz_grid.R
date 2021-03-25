data <- read.csv("grid.csv", header = T)

library(ggplot2)
library(ggthemes)

# Heatmap 
ggplot(data, aes(weight_decay, learning_rate, fill= bce_loss)) + 
  geom_tile() +
  scale_fill_gradient2(low = "green", mid = "yellow", high = "red", midpoint = 0.5) +
  theme
