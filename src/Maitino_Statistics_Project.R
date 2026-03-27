############################################################
# Project:   01 ALE
# Course:    Statistics for Data Science
# Master Degree: Data Science and Business Informatics
# University: University of Pisa
#
# File:      Maitino_Statistics_Project.R
# Author:    Tommaso Maitino
# Student ID: 582479
#
# Description:
#   This script contains personal code and experiments 
#   developed for the course "Statistics for Data Science".
#   The purpose is to reimplement some methods and experiments of the paper 
#   "Daniel W. Apley, Jingyu Zhu, Visualizing the Effects of Predictor Variables
#   in Black Box Supervised Learning Models, Journal of the Royal Statistical Society
#   Series B: Statistical Methodology, Volume 82, Issue 4, September 2020, Pages 1059–1086."
#   
#
# Created:   21/08/2025
# Last Update: 05/09/2025 
############################################################

# Load the necessary packages
library(ALEPlot)
library(tree)
library(nnet)
library(caret)

#Save default graphic layout parameters
old_par <- par(no.readonly = TRUE)

################### EXPERIMENT 1 ###########################

#Generate synthetic data
set.seed(42)
n <- 200
K_1 <- 50

x <- runif(n, min = 0, max = 1)
x1 <- x + rnorm(n, 0, 0.05)
x2 <- x + rnorm(n, 0, 0.05)
y <- x1 + x2^2
df <- data.frame(y, x1, x2)

#-----Figures difference PD and M plots computation
par(pty = "s")

#PD
plot(x1, x2, pch = 16, cex = 0.6, col = "black",
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = expression(italic(x[1])), ylab = expression(italic(x[2])),
     cex.lab = 1.5)

#Vertical line
abline(v = 0.3, col = "black", lwd = 2)

#Marginal density (approximated)
x2_density <- density(x2, adjust = 3)
curve_x <-  0.3 - x2_density$y * 0.15 
curve_y <- x2_density$x
valid_idx <- curve_y >= 0 & curve_y <= 1
lines(curve_x[valid_idx], curve_y[valid_idx], lwd = 2)

#M 
plot(x1, x2, pch = 16, cex = 0.6, col = "black",
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = expression(italic(x[1])), ylab = expression(italic(x[2])),
     cex.lab = 1.5)

#Vertical line
abline(v = 0.3, col = "black", lwd = 2)

#Conditional density (approximated)
#Select points near to x1 = 0.3
near_points <- abs(x1 - 0.3) < 0.05 #logic
if (sum(near_points) > 2) {
  x2_cond <- x2[near_points]
  x2_cond_density <- density(x2_cond)
  curve_x_cond <- 0.3 + x2_cond_density$y * 0.12
  curve_y_cond <- x2_cond_density$x
  valid_idx_cond <- curve_y_cond >= 0 & curve_y_cond <= 1
  lines(curve_x_cond[valid_idx_cond], curve_y_cond[valid_idx_cond], lwd = 2)
}

#Reset graphic layout
par(old_par)

#----
#fit the regression tree
tree_model <- tree(y ~ x1 + x2, data = df, control = tree.control(nobs = n, mindev = 0, minsize = 2), model=TRUE)
n_leaf_nodes <- length(tree_model$frame$var[tree_model$frame$var == "<leaf>"])
cat("Number of leaf nodes: ", n_leaf_nodes)

#Cross validation for choosing optimal number of leaf nodes

cv_results <- cv.tree(tree_model, FUN = prune.tree)

#Table with tree size and estimated deviance
results_table <- data.frame(
  LeafNodes = cv_results$size,
  Deviance = cv_results$dev
)
print(results_table)


plot(cv_results$size, cv_results$dev, type = "b",
     xlab = "Number of leaf nodes",
     ylab = "Deviance",
     main = "Cross-validation for choosing the tree size")

#Find minimum deviance
min_dev <- min(cv_results$dev)

#Find the sizes that achieve minimum deviance
best_sizes <- cv_results$size[cv_results$dev == min_dev]

#Select the minimum size
best_size <- min(best_sizes)

cat("Optimal number of leaf nodes:", best_size, "\n")

#Prune the tree 
pruned_tree <- prune.tree(tree_model, best = best_size)

#Define the predictive function
yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata, type= "vector"))

#Calculate and plot the ALE and PD main effects of x1 and x2
par(mfrow = c(2,2), mar = c(4,4,2,2) + 0.1, pty = "s")
ALE.1 = ALEPlot(df[,2:3], pruned_tree, pred.fun = yhat, J = 1, K = K_1, NA.plot = TRUE)
PD.1 = PDPlot(df[,2:3], pruned_tree, pred.fun = yhat, J = 1, K = K_1)
ALE.2 = ALEPlot(df[,2:3], pruned_tree, pred.fun = yhat, J = 2, K = K_1, NA.plot = TRUE)
PD.2 = PDPlot(df[,2:3], pruned_tree, pred.fun = yhat, J = 2, K = K_1)

par(old_par)
#M-Plot implementation -------------------------------------------------

#Function to calculate M-plot
calculate_m_plot <- function(model, data, var_name, n_points = 50, bandwidth = 0.05) {
  
  #Variable range
  var_range <- seq(min(data[[var_name]]), max(data[[var_name]]), 
                   length.out = n_points)
  
  baseline_pred <- mean(predict(model, data))
  
  m_effects <- sapply(var_range, function(x_target) {
    
    #Find N(x_target) neighborhood 
    neighborhood_indices <- which(abs(data[[var_name]] - x_target) <= bandwidth)
    n_x <- length(neighborhood_indices) 
    
    if(n_x == 0) {
      #If there are no observations in the neighborhood, expand the bandwidth.
      bandwidth_expanded <- bandwidth * 2
      neighborhood_indices <- which(abs(data[[var_name]] - x_target) <= bandwidth_expanded)
      n_x <- length(neighborhood_indices)
    }
    
    if(n_x == 0) return(NA)
    
    #Neighborhood observations
    neighborhood_data <- data[neighborhood_indices, ]
    
    #Create the dataset for the prediction: fix the target variable, keep the others from the neighborhood
    pred_data <- neighborhood_data
    pred_data[[var_name]] <- x_target
    
    predictions <- predict(model, pred_data)
    
    # Mean: (1/n(x)) * sum(f(x_target, x_other_from_neighborhood))
    mean(predictions) - baseline_pred
  })
  
  data.frame(x = var_range, effect = m_effects)
}

#Calculate M-Plots for x1 and x2

#M-Plot for x1
m_plot_x1 <- calculate_m_plot(pruned_tree, df, "x1", n_points = 50, bandwidth = 0.03)

#M-Plot for x2  
m_plot_x2 <- calculate_m_plot(pruned_tree, df, "x2", n_points = 50, bandwidth = 0.03)

#Remove NA
m_plot_x1 <- m_plot_x1[!is.na(m_plot_x1$effect), ]
m_plot_x2 <- m_plot_x2[!is.na(m_plot_x2$effect), ]

# PDF saving block (uncomment the next line and the command "dev.off()" to save the graphs image)
# pdf("C:\\Users\\Tommaso\\Documents\\Università\\Magistrale\\Primo anno (5 su 8)\\Secondo semestre (3 su 5)\\Statistics for data science\\Stat_project\\Images\\Maitino_ALE_Exp_1.pdf", width = 20, height = 10)

#Graphic layout
par(mfrow = c(1, 2), mar = c(4, 4, 3, 2))

#M-Plot of x1
plot(m_plot_x1$x, m_plot_x1$effect, 
     type = "l",
     lty = 2,
     lwd = 2,
     xlab = expression(italic(x[1])), 
     ylab = "Estimated effect",
     xlim = c(0, 1),
     ylim = c(-0.7, 0.7),
     cex.lab = 1.3)

#Add ALE Plot of x1
lines(ALE.1$x.values, ALE.1$f.values, col = "blue", lwd = 2)

points(ALE.1$x.values, ALE.1$f.values, col = "blue", pch = 16, cex = 0.8)

#Add PD Plot of x1
lines(PD.1$x.values, PD.1$f.values, col = "red", lwd = 2, lty = 2)

#Add true effect of x1
curve(x - 0.5, from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#M-Plot of x2
plot(m_plot_x2$x, m_plot_x2$effect, 
     type = "l",
     lty = 2,
     lwd = 2,
     xlab = expression(italic(x[2])), 
     ylab = "Estimated effect",
     xlim = c(0, 1),
     ylim = c(-0.7, 0.7),
     cex.lab = 1.3)

#Add ALE Plot of x2
lines(ALE.2$x.values, ALE.2$f.values, col = "blue", lwd = 2)

points(ALE.2$x.values, ALE.2$f.values, col = "blue", pch = 16, cex = 0.8)

#Add PD Plot of x2
lines(PD.2$x.values, PD.2$f.values, col = "red", lwd = 2, lty = 2)

#Add true effect of x2
curve(x^2 - (1/3 + 0.05^2), from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#Reset graphic layout
par(old_par)

# dev.off()

##################### RE-IMPLEMENTATION WITH PAPER HYPERPARAMETERS #############################

#Prune the tree to 100 leaf nodes as in the paper
pruned_tree_paper <- prune.tree(tree_model, best = 100)

pruned_tree_plot <- prune.tree(tree_model, best = 8)

#Save the original frame
original_frame <- pruned_tree_plot$frame

#Function to round numbers in split strings
round_splits <- function(split_string, digits = 3) {
  if(is.na(split_string) || split_string == "") return(split_string)
  
  #Extract the number from the string
  number <- as.numeric(gsub("[<>]", "", split_string))
  #Round and reconstruct the string
  operator <- ifelse(grepl("<", split_string), "<", ">")
  return(paste0(operator, round(number, digits)))
}

#Apply rounding to the splits matrix
#First column (cutleft)
pruned_tree_plot$frame$splits[, 1] <- sapply(pruned_tree_plot$frame$splits[, 1], 
                                             function(x) round_splits(x, 3))

#Second column (cutright)  
pruned_tree_plot$frame$splits[, 2] <- sapply(pruned_tree_plot$frame$splits[, 2], 
                                             function(x) round_splits(x, 3))

#Plot the tree
plot(pruned_tree_plot)
text(pruned_tree_plot, digits = getOption("digits") - 5, pretty = TRUE, cex = 0.8)
title(paste("Tree pruned at", 8, "leaf nodes"))

#Restore the original frame if necessary:
pruned_tree_plot$frame <- original_frame

#x1-x2 scatter plot
plot(df$x1, df$x2, pch = 19, cex = 0.8, xlab = expression(italic(x[1])), ylab = expression(italic(x[2])), cex.lab = 1.3)

#Filter the splits to add them to the graph
splits <- pruned_tree_paper$frame[pruned_tree_paper$frame$var != "<leaf>", ]

#Sort splits by node ID (numeric row names)
splits$node_id <- as.numeric(rownames(splits))
splits <- splits[order(splits$node_id), ]

split_values <- as.numeric(gsub("[<>]", "", splits$splits[, "cutleft"]))

#Show splits together with variables
data.frame(
  node_id = splits$node_id,
  variabile = splits$var, 
  valore_split = split_values
)

n_splits_to_show <- min(7, nrow(splits))

#Add the splits in order of node ID
for(i in 1:n_splits_to_show) {
  split_var <- as.character(splits$var[i])
  node_id <- splits$node_id[i]
  
  split_value_text <- splits$splits[i, "cutleft"]
  split_value <- as.numeric(gsub("[<>]", "", split_value_text))
  
  cat("Split", i, "(Node", node_id, "):", split_var, "<", split_value, "\n")
  
  if(split_var == "x1") {
    abline(v = split_value, lwd = 2)
  } else if(split_var == "x2") {
    abline(h = split_value, lwd = 2)
  }
}

#Reset graphic layout
par(old_par)

#Calculate and plot the ALE and PD main effects of x1 and x2
par(mfrow = c(2,2), mar = c(4,4,2,2) + 0.1, pty = "s")
ALE.1 = ALEPlot(df[,2:3], pruned_tree_paper, pred.fun = yhat, J = 1, K = K_1, NA.plot = TRUE)
PD.1 = PDPlot(df[,2:3], pruned_tree_paper, pred.fun = yhat, J = 1, K = K_1)
ALE.2 = ALEPlot(df[,2:3], pruned_tree_paper, pred.fun = yhat, J = 2, K = K_1, NA.plot = TRUE)
PD.2 = PDPlot(df[,2:3], pruned_tree_paper, pred.fun = yhat, J = 2, K = K_1)

#Reset graphic layout
par(old_par)

#Calculate M-Plots for x1 and x2

#M-Plot for x1
m_plot_x1 <- calculate_m_plot(pruned_tree_paper, df, "x1", n_points = 50, bandwidth = 0.03)

#M-Plot for x2  
m_plot_x2 <- calculate_m_plot(pruned_tree_paper, df, "x2", n_points = 50, bandwidth = 0.03)

#Remove NA
m_plot_x1 <- m_plot_x1[!is.na(m_plot_x1$effect), ]
m_plot_x2 <- m_plot_x2[!is.na(m_plot_x2$effect), ]

#PDF saving block (uncomment the next line and the command "dev.off()" to save the graphs image)
# pdf("C:\\Users\\Tommaso\\Documents\\Università\\Magistrale\\Primo anno (5 su 8)\\Secondo semestre (3 su 5)\\Statistics for data science\\Stat_project\\Images\\Maitino_ALE_Exp_1_paper.pdf", width = 20, height = 10)

#Graphic layout
par(mfrow = c(1, 2), mar = c(4, 4, 3, 2))

#M-Plot of x1
plot(m_plot_x1$x, m_plot_x1$effect, 
     type = "l",
     lty = 2,
     lwd = 2,
     xlab = expression(italic(x[1])), 
     ylab = "Estimated effect",
     xlim = c(0, 1),
     ylim = c(-0.7, 0.7),
     cex.lab = 1.3)

#Add ALE Plot of x1
lines(ALE.1$x.values, ALE.1$f.values, col = "blue", lwd = 2)

points(ALE.1$x.values, ALE.1$f.values, col = "blue", pch = 16, cex = 0.8)

#Add PD Plot of x1
lines(PD.1$x.values, PD.1$f.values, col = "red", lwd = 2, lty = 2)

#Add true effect of x1
curve(x - 0.5, from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#M-Plot of x2
plot(m_plot_x2$x, m_plot_x2$effect, 
     type = "l",
     lty = 2,
     lwd = 2,
     xlab = expression(italic(x[2])), 
     ylab = "Estimated effect",
     xlim = c(0, 1),
     ylim = c(-0.7, 0.7),
     cex.lab = 1.3)

#Add ALE Plot of x2
lines(ALE.2$x.values, ALE.2$f.values, col = "blue", lwd = 2)

points(ALE.2$x.values, ALE.2$f.values, col = "blue", pch = 16, cex = 0.8)

#Add PD Plot of x2
lines(PD.2$x.values, PD.2$f.values, col = "red", lwd = 2, lty = 2)

#Add true effect of x2
curve(x^2 - (1/3 + 0.05^2), from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#Reset graphic layout
par(old_par)

# dev.off()

################### EXPERIMENT 2 #######################

#Set parameters
n <- 200 
n_simulations <- 50
K_2 = 50

#Set seed for reproducibility
set.seed(123)

#---PHASE 1: HYPERPARAMETER TUNING VIA REPEATED K-FOLD CROSS-VALIDATION

#Generate initial dataset
x <- runif(n, min = 0, max = 1)
x1 <- x + rnorm(n, 0, 0.05)
x2 <- x + rnorm(n, 0, 0.05)
y <- x1 + x2^2 + rnorm(n, 0, 0.1)
df_tuning <- data.frame(y, x1, x2)

#Calculate theoretical R²
var_epsilon <- 0.1^2  # variance of noise term
var_y <- var(y)       # variance of Y in the data
theoretical_r2 <- 1 - var_epsilon/var_y
cat("Theoretical R² =", round(theoretical_r2, 4), "\n\n")

#Define training control for repeated k-fold cross-validation
train_control <- trainControl(
  method = "repeatedcv",
  number = 10,           # k = 10 folds
  repeats = 5,           # 5 repetitions
  summaryFunction = defaultSummary,
  returnResamp = "all",
  savePredictions = "final"
)

#Define hyperparameters grid 
size_grid <- c(5, 8, 10, 12, 15)
decay_grid <- c(0.0001, 0.001, 0.01, 0.1)

#Create tuning grid
tune_grid <- expand.grid(
  size = size_grid,
  decay = decay_grid
)

cat("Performing hyperparameter search with repeated k-fold cross-validation...\n")
cat("Grid search: ", nrow(tune_grid), " parameter combinations\n")
cat("CV setup: 10-fold repeated 5 times = 50 total CV runs per combination\n\n")

#Train model with hyperparameter tuning
nnet_model <- train(
  y ~ ., 
  data = df_tuning,
  method = "nnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  linout = TRUE,
  skip = FALSE,
  maxit = 1000,
  trace = FALSE
)

#Extract best parameters and results
best_params <- nnet_model$bestTune
best_r2 <- max(nnet_model$results$Rsquared)
r2_range <- range(nnet_model$results$Rsquared)

cat("=== CROSS-VALIDATION RESULTS ===\n")
cat("Best hyperparameters: size =", best_params$size, ", decay =", best_params$decay, "\n")
cat("Best CV R² =", round(best_r2, 4), "\n\n")
cat("Range CV R² =", round(r2_range, 4), "\n\n")

#----PHASE 2: MONTE CARLO SIMULATIONS WITH OPTIMAL HYPERPARAMETERS

#Reset seed for Monte Carlo simulations
set.seed(42)

#Initialize storage for results
ALE_results_x1 <- list()
PD_results_x1 <- list()
ALE_results_x2 <- list()
PD_results_x2 <- list()

#Define the predictive function
yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata, type= "raw"))

#Monte Carlo simulations with optimal hyperparameters
for (i in 1:n_simulations) {
  cat("Running simulation", i, "of", n_simulations, "\n")
  
  #Generate synthetic data for this iteration
  x <- runif(n, min = 0, max = 1)
  x1 <- x + rnorm(n, 0, 0.05)
  x2 <- x + rnorm(n, 0, 0.05)
  y = x1 + x2^2 + rnorm(n, 0, 0.1)
  df = data.frame(y, x1, x2)
  
  #Fit neural network with optimal hyperparameters
  nnet.df <- nnet(y ~ ., data = df, linout = TRUE, skip = FALSE, 
                  size = best_params$size, decay = best_params$decay, 
                  maxit = 1000, trace = FALSE)
  
  #Calculate ALE and PD plots
  ALE.1 = ALEPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 1, K = K_2, NA.plot = TRUE)
  PD.1 = PDPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 1, K = K_2)
  ALE.2 = ALEPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 2, K = K_2, NA.plot = TRUE)
  PD.2 = PDPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 2, K = K_2)
  
  #Store results
  ALE_results_x1[[i]] <- list(x = ALE.1$x.values, f = ALE.1$f.values)
  PD_results_x1[[i]] <- list(x = PD.1$x.values, f = PD.1$f.values)
  ALE_results_x2[[i]] <- list(x = ALE.2$x.values, f = ALE.2$f.values)
  PD_results_x2[[i]] <- list(x = PD.2$x.values, f = PD.2$f.values)
}

#-----PHASE 3: VISUALIZATION
# PDF saving block (uncomment the next line and the command "dev.off()" to save the graphs image)
# pdf("C:\\Users\\Tommaso\\Documents\\Università\\Magistrale\\Primo anno (5 su 8)\\Secondo semestre (3 su 5)\\Statistics for data science\\Stat_project\\Images\\Maitino_ALE_MonteCarlo.pdf", width = 12, height = 12)

#Set up the plot layout
par(mfrow = c(2,2), mar = c(5, 5, 4, 2), cex.lab = 1.4, cex.main = 1.4, cex.axis = 1.3, pty = "s")

#Plot ALE for x1
plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
     xlab = expression(italic(x[1])), 
     ylab = expression(italic(hat(f)[1*","*ALE] * "(x"[1]*")")), 
     main = expression("(a) ALE Plot for x"[1]))

#Plot all Monte Carlo results in light gray
for (i in 1:n_simulations) {
  x_vals <- ALE_results_x1[[i]]$x
  f_vals <- ALE_results_x1[[i]]$f
  valid_idx <- x_vals >= 0 & x_vals <= 1
  lines(x_vals[valid_idx], f_vals[valid_idx], col = "lightgray", lwd = 0.5)
}

#Plot true effect in black
curve(x - 0.5, from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#Plot PD for x1
plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
     xlab = expression(italic(x[1])), 
     ylab = expression(italic(hat(f)[1*","*PD] * "(x"[1]*")")), 
     main = expression("(b) PD Plot for x"[1]))

#Plot all Monte Carlo results in light gray
for (i in 1:n_simulations) {
  x_vals <- PD_results_x1[[i]]$x
  f_vals <- PD_results_x1[[i]]$f
  valid_idx <- x_vals >= 0 & x_vals <= 1
  lines(x_vals[valid_idx], f_vals[valid_idx], col = "lightgray", lwd = 0.5)
}

#Plot true effect in black
curve(x - 0.5, from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#Plot ALE for x2
plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
     xlab = expression(italic(x[2])), 
     ylab = expression(italic(hat(f)[2*","*ALE] * "(x"[2]*")")), 
     main = expression("(c) ALE Plot for x"[2]))

#Plot all Monte Carlo results in light gray
for (i in 1:n_simulations) {
  x_vals <- ALE_results_x2[[i]]$x
  f_vals <- ALE_results_x2[[i]]$f
  valid_idx <- x_vals >= 0 & x_vals <= 1
  lines(x_vals[valid_idx], f_vals[valid_idx], col = "lightgray", lwd = 0.5)
}

#Plot true effect in black
curve(x^2 - (1/3+0.05^2), from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#Plot PD for x2
plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
     xlab = expression(italic(x[2])), 
     ylab = expression(italic(hat(f)[2*","*PD] * "(x"[2]*")")), 
     main = expression("(d) PD Plot for x"[2]))

#Plot all Monte Carlo results in light gray
for (i in 1:n_simulations) {
  x_vals <- PD_results_x2[[i]]$x
  f_vals <- PD_results_x2[[i]]$f
  valid_idx <- x_vals >= 0 & x_vals <= 1
  lines(x_vals[valid_idx], f_vals[valid_idx], col = "gray", lwd = 0.5)
}

#Plot true effect in black
curve(x^2 - (1/3+0.05^2), from = 0, to = 1, add = TRUE, col = "black", lwd = 2)

#Reset graphic layout
par(old_par)

# dev.off()

cat("Monte Carlo analysis completed with", n_simulations, "simulations\n")

#Print final summary
cat("\n=== FINAL SUMMARY ===\n")
cat("Optimal hyperparameters: size =", best_params$size, ", decay =", best_params$decay, "\n")
cat("CV R² range: [",round(r2_range,3),"]\n")
cat("Theoretical R²: ", round(theoretical_r2, 3), "\n")
cat("Paper R² range: [0.965, 0.975]\n")
cat("Analysis completed!\n")

#Additional simulation with paper's hyperparameters if different from optimal ones
if (best_params$size != 10 || best_params$decay != 0.0001) {
  
  cat("\n=== ADDITIONAL SIMULATION WITH PAPER'S HYPERPARAMETERS ===\n")
  cat("Optimal parameters differ from paper's parameters (size=10, decay=0.0001)\n")
  cat("Running additional Monte Carlo simulation with paper's hyperparameters...\n\n")
  
  #Reset seed for reproducibility
  set.seed(42)
  
  #Initialize storage for paper's results
  ALE_results_x1_paper <- list()
  PD_results_x1_paper <- list()
  ALE_results_x2_paper <- list()
  PD_results_x2_paper <- list()
  
  #Monte Carlo simulations with paper's hyperparameters
  for (i in 1:n_simulations) {
    cat("Running paper simulation", i, "of", n_simulations, "\n")
    
    #Generate synthetic data for this iteration
    x <- runif(n, min = 0, max = 1)
    x1 <- x + rnorm(n, 0, 0.05)
    x2 <- x + rnorm(n, 0, 0.05)
    y = x1 + x2^2 + rnorm(n, 0, 0.1)
    df = data.frame(y, x1, x2)
    
    #Fit neural network with paper's hyperparameters
    nnet.df <- nnet(y ~ ., data = df, linout = TRUE, skip = FALSE, 
                    size = 10, decay = 0.0001, 
                    maxit = 1000, trace = FALSE)
    
    #Calculate ALE and PD plots
    ALE.1 = ALEPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 1, K = K_2, NA.plot = TRUE)
    PD.1 = PDPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 1, K = K_2)
    ALE.2 = ALEPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 2, K = K_2, NA.plot = TRUE)
    PD.2 = PDPlot(df[,2:3], nnet.df, pred.fun = yhat, J = 2, K = K_2)
    
    #Store results
    ALE_results_x1_paper[[i]] <- list(x = ALE.1$x.values, f = ALE.1$f.values)
    PD_results_x1_paper[[i]] <- list(x = PD.1$x.values, f = PD.1$f.values)
    ALE_results_x2_paper[[i]] <- list(x = ALE.2$x.values, f = ALE.2$f.values)
    PD_results_x2_paper[[i]] <- list(x = PD.2$x.values, f = PD.2$f.values)
  }
  
  cat("=== PAPER'S HYPERPARAMETERS VISUALIZATION ===\n")
  # PDF saving block (uncomment the next line and the command "dev.off()" to save the graphs image)
  # pdf("C:\\Users\\Tommaso\\Documents\\Università\\Magistrale\\Primo anno (5 su 8)\\Secondo semestre (3 su 5)\\Statistics for data science\\Stat_project\\Images\\Maitino_ALE_MonteCarlo_Paper.pdf", width = 12, height = 12)
  
  #Set up the plot layout
  par(mfrow = c(2,2), mar = c(5, 5, 4, 2), cex.lab = 1.4, cex.main = 1.4, cex.axis = 1.3, pty = "s")
  
  #Plot ALE for x1
  plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
       xlab = expression(italic(x[1])), 
       ylab = expression(italic(hat(f)[1*","*ALE] * "(x"[1]*")")), 
       main = expression("(a) ALE Plot for x"[1]))
  
  #Plot all Monte Carlo results in light gray
  for (i in 1:n_simulations) {
    x_vals <- ALE_results_x1_paper[[i]]$x
    f_vals <- ALE_results_x1_paper[[i]]$f
    valid_idx <- x_vals >= 0 & x_vals <= 1
    lines(x_vals[valid_idx], f_vals[valid_idx], col = "gray", lwd = 0.5)
  }
  
  #Plot true effect in black
  curve(x - 0.5, from = 0, to = 1, add = TRUE, col = "black", lwd = 2)
  
  #Plot PD for x1
  plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
       xlab = expression(italic(x[1])), 
       ylab = expression(italic(hat(f)[1*","*PD] * "(x"[1]*")")), 
       main = expression("(b) PD Plot for x"[1]))
  
  #Plot all Monte Carlo results in light gray
  for (i in 1:n_simulations) {
    x_vals <- PD_results_x1_paper[[i]]$x
    f_vals <- PD_results_x1_paper[[i]]$f
    valid_idx <- x_vals >= 0 & x_vals <= 1
    lines(x_vals[valid_idx], f_vals[valid_idx], col = "lightgray", lwd = 0.5)
  }
  
  #Plot true effect in black
  curve(x - 0.5, from = 0, to = 1, add = TRUE, col = "black", lwd = 2)
  
  #Plot ALE for x2
  plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
       xlab = expression(italic(x[2])), 
       ylab = expression(italic(hat(f)[2*","*ALE] * "(x"[2]*")")), 
       main = expression("(c) ALE Plot for x"[2]))
  
  #Plot all Monte Carlo results in light gray
  for (i in 1:n_simulations) {
    x_vals <- ALE_results_x2_paper[[i]]$x
    f_vals <- ALE_results_x2_paper[[i]]$f
    valid_idx <- x_vals >= 0 & x_vals <= 1
    lines(x_vals[valid_idx], f_vals[valid_idx], col = "lightgray", lwd = 0.5)
  }
  
  #Plot true effect in black
  curve(x^2 - (1/3+0.05^2), from = 0, to = 1, add = TRUE, col = "black", lwd = 2)
  
  #Plot PD for x2
  plot(NULL, xlim = c(0,1), ylim = c(-1,1), 
       xlab = expression(italic(x[2])), 
       ylab = expression(italic(hat(f)[2*","*PD] * "(x"[2]*")")), 
       main = expression("(d) PD Plot for x"[2]))
  
  #Plot all Monte Carlo results in light gray
  for (i in 1:n_simulations) {
    x_vals <- PD_results_x2_paper[[i]]$x
    f_vals <- PD_results_x2_paper[[i]]$f
    valid_idx <- x_vals >= 0 & x_vals <= 1
    lines(x_vals[valid_idx], f_vals[valid_idx], col = "lightgray", lwd = 0.5)
  }
  
  #Plot true effect in black
  curve(x^2 - (1/3+0.05^2), from = 0, to = 1, add = TRUE, col = "black", lwd = 2)
  
  # dev.off()
  
  cat("Paper's hyperparameters Monte Carlo analysis completed with", n_simulations, "simulations\n")
  
  #Print comparison summary
  cat("\n=== COMPARISON SUMMARY ===\n")
  cat("OPTIMAL HYPERPARAMETERS (Cross-Validation):\n")
  cat("  size =", best_params$size, ", decay =", best_params$decay, "\n")
  cat("Range CV R² =", round(r2_range, 4), "\n\n")
  cat("PAPER'S HYPERPARAMETERS:\n")
  cat("  size = 10, decay = 0.0001\n")
  cat("Theoretical R²: ", round(theoretical_r2, 3), "\n")
  cat("Additional analysis completed!\n")
  
} else {
  cat("\n=== NO ADDITIONAL SIMULATION NEEDED ===\n")
  cat("Optimal hyperparameters match paper's parameters (size=10, decay=0.0001)\n")
  cat("No additional simulation required.\n")
}

################### EXPERIMENT 3 #######################

#Load data
data = read.csv("C:\\Users\\Tommaso\\Documents\\bike+sharing+dataset\\hour.csv")

#Remove unnecessary columns
df <- data[, !names(data) %in% c("instant", "dteday", "season", "casual", "registered")]

d=cor(df)
d[abs(d) < 0.8] = NA
View(d)

#Create a backup of the column "cnt" to be used later for normalization
cnt_backup <- df$cnt

#Save original max and min of the target variable
original_min <- min(df$cnt)
original_max <- max(df$cnt)

#Normalize (min-max) the target variable
df$cnt <- (df$cnt - original_min) / (original_max - original_min)

#Set parameter K
K_3 = 100

#Set seed for reproducibility
set.seed(1)

#Define training control for repeated k-fold cross-validation
train_control <- trainControl(
  method = "repeatedcv",
  number = 3,           # k = 3 folds
  repeats = 5,           # 5 repetitions
  summaryFunction = defaultSummary,
  returnResamp = "all",
  savePredictions = "final",
  verboseIter = TRUE
)

#Define hyperparameters grid
size_grid <- c(5, 8, 10, 12, 15)
decay_grid <- c(0.0001, 0.001, 0.01, 0.05, 0.1)

#Create tuning grid
tune_grid <- expand.grid(
  size = size_grid,
  decay = decay_grid
)

cat("Performing hyperparameter search with repeated k-fold cross-validation...\n")
cat("Grid search: ", nrow(tune_grid), " parameter combinations\n")
cat("CV setup: 3-fold repeated 5 times = 15 total CV runs per combination\n\n")

#(VERY SLOW) Train model with hyperparameters tuning
nnet_model <- train(
  cnt ~ .,
  data = df,
  method = "nnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  linout = FALSE,
  skip = FALSE,
  maxit = 1000,
  trace = FALSE
)

#Extract best parameters and results
best_params <- nnet_model$bestTune
best_r2 <- max(nnet_model$results$Rsquared)

cat("=== CROSS-VALIDATION RESULTS ===\n")
cat("Best hyperparameters: size =", best_params$size, ", decay =", best_params$decay, "\n")
cat("Best CV R² =", round(best_r2, 4), "\n\n")


#ANALYSIS TO BE PERFORMED BOTH WITH THE HYPERPARAMETERS FOUND THROUGH CROSS-VALIDATION
#AND WITH THOSE USED IN THE PAPER

#Cross-validation hyperparameters (size = 15, decay = 0.001)
cv_size <- best_params$size
cv_decay <- best_params$decay

#Paper's hyperparameters
paper_size <- 10
paper_decay <- 0.05

#Fit neural network with paper's hyperparameters
nnet.df <- nnet(cnt ~ ., data = df, linout = F, skip = F, size = paper_size, decay = paper_decay, maxit = 1000, trace = F,  MaxNWts = 1000)

#Define the predictive function
yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata, type="raw"))

#Calculate and plot the ALE main effects
ALE.2 = ALEPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat, J=2, K=K_3, NA.plot = TRUE)
ALE.3 = ALEPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat, J=3, K=K_3, NA.plot = TRUE)
ALE.7 = ALEPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat, J=7, K=K_3, NA.plot = TRUE)
ALE.11 = ALEPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat, J=11, K=K_3, NA.plot = TRUE)

#Shows the time taken to generate the graphs of the main effect
time_ALE <- system.time({
  ALE.9 = ALEPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat, J=9, K=K_3, NA.plot = TRUE)
})
print(time_ALE)


time_PD <- system.time({
  PD.9 = PDPlot(df[,-ncol(df)], nnet.df, pred.fun = yhat, J = 9, K = K_3)
})
print(time_PD)

#Calculate the zero order effect (expected value)
all_predictions <- predict(nnet.df, df[,-ncol(df)], type = "raw")
zero_order_effect <- mean(all_predictions)

#Denormalize the target variable for the plot
df$cnt <- cnt_backup

zero_order_effect <- zero_order_effect * (original_max - original_min) + original_min

cat("Zero order effect (E[f(X)]):", round(zero_order_effect, 2), "\n")

ALE.2$f.values <- ALE.2$f.values * (original_max - original_min) + original_min
ALE.3$f.values <- ALE.3$f.values * (original_max - original_min) + original_min
ALE.7$f.values <- ALE.7$f.values * (original_max - original_min) + original_min
ALE.11$f.values <- ALE.11$f.values * (original_max - original_min) + original_min

ALE.9$f.values <- ALE.9$f.values * (original_max - original_min) + original_min
PD.9$f.values <- PD.9$f.values * (original_max - original_min) + original_min

max_atemp = 50
min_atemp = -16
ALE.9$x.values <- ALE.9$x.values * (max_atemp - min_atemp) + min_atemp
PD.9$x.values <- PD.9$x.values * (max_atemp - min_atemp) + min_atemp

# PDF saving block (uncomment the next line and the command "dev.off()" to save the graphs image)
# pdf("C:\\Users\\Tommaso\\Documents\\Università\\Magistrale\\Primo anno (5 su 8)\\Secondo semestre (3 su 5)\\Statistics for data science\\Stat_project\\Images\\Maitino_ALE_1_Exp_3_paper.pdf", width = 20, height = 10)

par(mfrow = c(2,2), mar = c(4,5,2,2) + 0.1)
plot(ALE.2$x.values, ALE.2$f.values + zero_order_effect, type="l", xlab = expression("month (" * italic(x[2]) * ")"), ylab = expression(italic(hat(f)[2*","*ALE] * "(x"[2]*")")), main = "(a)", cex.lab = 1.3)
plot(ALE.3$x.values, ALE.3$f.values + zero_order_effect, type="l", xlab = expression("hour (" * italic(x[3]) * ")"), ylab = expression(italic(hat(f)[3*","*ALE] * "(x"[3]*")")), main = "(b)", cex.lab = 1.3)
plot(ALE.7$x.values, ALE.7$f.values + zero_order_effect, type="l", xlab = expression("weather situation (" * italic(x[7]) * ")"), ylab = expression(italic(hat(f)[7*","*ALE] * "(x"[7]*")")), main = "(c)", cex.lab = 1.3)
plot(ALE.11$x.values, ALE.11$f.values + zero_order_effect, type="l", xlab = expression("wind speed (" * italic(x[11]) * ")"), ylab = expression(italic(hat(f)[11*","*ALE] * "(x"[11]*")")), main = "(d)", cex.lab = 1.3)

# dev.off()

#Reset graphic layout
par(old_par)

# PDF saving block (uncomment the next line and the command "dev.off()" to save the graphs image)
# pdf("C:\\Users\\Tommaso\\Documents\\Università\\Magistrale\\Primo anno (5 su 8)\\Secondo semestre (3 su 5)\\Statistics for data science\\Stat_project\\Images\\Maitino_ALE_2_Exp_3_paper.pdf", width = 20, height = 10)

par(mfrow = c(1,2), mar = c(4,5,2,2) + 0.1)
plot(ALE.9$x.values, ALE.9$f.values + zero_order_effect, type="l", xlab = expression("feeling temperature (" * italic(x[9]) * ")"), ylab = expression(italic(hat(f)[9*","*ALE] * "(x"[9]*")")), main = "(a)", cex.lab = 1.3)
plot(PD.9$x.values, PD.9$f.values + zero_order_effect, type="l", xlab = expression("feeling temperature (" * italic(x[9]) * ")"), ylab = expression(italic(hat(f)[9*","*PD] * "(x"[9]*")")), main = "(b)", cex.lab = 1.3)

# dev.off()

#Reset graphic layout
par(old_par)

############## INTERACTION PLOTS ###############################

#create a new predict function for the normalized target variable
yhat_denorm <- function(X.model, newdata) {
  #Get normalized predictions using the same structure as yhat
  pred_norm <- as.numeric(predict(X.model, newdata, type="raw"))
  
  #De-normalize 
  pred_orig <- pred_norm * (original_max - original_min) + original_min
  
  return(pred_orig)
}


ALE.3and7 <- ALEPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat_denorm, J=c(3,7), K=K_3, NA.plot = TRUE)

# PDF saving block (uncomment the next line and the command "dev.off()" to save the graphs image)
# pdf("C:\\Users\\Tommaso\\Documents\\Università\\Magistrale\\Primo anno (5 su 8)\\Secondo semestre (3 su 5)\\Statistics for data science\\Stat_project\\Images\\Maitino_ALE_3_Exp_3_paper.pdf", width = 20, height = 10)
par(mfrow = c(1,2), mar = c(4,5,2,2) + 0.1)
image(ALE.3and7$x.values[[1]], ALE.3and7$x.values[[2]], ALE.3and7$f.values, col=heat.colors(20), 
      xlab = expression(hour~(italic(x[3]))),
      ylab = expression(weather~situation~(italic(x[7]))),
      main="(a)",
      ylim = c(1, 4),
      cex.lab = 2.2)
contour(ALE.3and7$x.values[[1]], ALE.3and7$x.values[[2]], ALE.3and7$f.values, add=TRUE, labcex = 1.5)

#Create the matrix of the main combined effects
main_effects_matrix <- outer(ALE.3$f.values, ALE.7$f.values, "+")

#Add to the interaction effect
ALE.3and7_zero <- ALE.3and7$f.values + main_effects_matrix + zero_order_effect

image(ALE.3and7$x.values[[1]], ALE.3and7$x.values[[2]], ALE.3and7_zero, col=heat.colors(20), 
      xlab = expression(hour~(italic(x[3]))),
      ylab = expression(weather~situation~(italic(x[7]))),
      main="(a)",
      ylim = c(1, 4),
      cex.lab = 2.2)
contour(ALE.3and7$x.values[[1]], ALE.3and7$x.values[[2]], ALE.3and7_zero, add=TRUE, labcex = 1.5)

# dev.off()

#Reset graphic layout
par(old_par)

#Show the time taken to generate the graphs of the interaction effect
time_ALE_int <- system.time({
  ALE.3and7 <- ALEPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat_denorm, J=c(3,7), K=K_3, NA.plot = TRUE)
})
print(time_ALE_int)

#5/6 minutes
time_PD_int <- system.time({
  PD.3and7 <- PDPlot(df[,-ncol(df)], nnet.df, pred.fun=yhat_denorm, J=c(3,7), K=K_3)
})
print(time_PD_int)
#################################    END    ######################################