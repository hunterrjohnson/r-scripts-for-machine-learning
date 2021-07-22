# Random forest model using grid search w/ ranger package

lapply(c('dplyr','data.table','mlr','ranger'), library, character.only = TRUE)

setwd('')

# Read data
dat <- data.frame(fread('example.csv'))

# Specify y and x variables for models
y_cols <- c()
x_cols <- c()

# Convert dependent variables to factor
dat[y_cols] <- lapply(dat[, y_cols], as.factor)

dat <- dat[, c(y_cols, x_cols) ]
rm(x_cols)

#===============================================================================
# Random forest with grid search

# Hyperparameters for grid search
hyper_grid <- expand.grid(
  mtry       = seq(2, 14, by = 1),
  node_size  = seq(1, 13, by = 4),
  samp_size = c(.625, .75, .875)
)

# Grid search RF function
grid_search_rf <- function(y_var, dat, grid){
  
  # Subset to non-missing variables
  dat <- dat[which(!is.na(dat[[y_var]])), ]
  
  # Train/test split
  set.seed(123)
  dat_split <- initial_split(dat, prop = .75)
  dat_train <- training(dat_split)
  dat_test  <- testing(dat_split)
  
  for(i in 1:nrow(grid)) {
    
    # Print current iteration
    cat(paste0('Iteration: ', i, ' of ', nrow(grid)), '\n')
    
    # Train model
    model <- ranger(
      formula         = dat_train[[y_var]]~var1+var2+var3+var4+var5,
      data            = dat_train, 
      classification  = TRUE,
      num.trees       = 500,
      mtry            = grid$mtry[i],
      min.node.size   = grid$node_size[i],
      sample.fraction = grid$samp_size[i],
      seed            = 123
    )
    
    # Add OOB metrics to grid
    grid$oob_accuracy[i] <- (1 - model$prediction.error)
    grid$oob_sensitivity[i] <- (model$confusion.matrix[1,1] / sum(model$confusion.matrix[1,]))
    grid$oob_specificity[i] <- (model$confusion.matrix[2,2] / sum(model$confusion.matrix[2,]))
    
    # Predict on test data
    pred <- predict(model, dat_test)
    dat_test$pred <- pred$predictions
    
    # Add test metrics to grid
    grid$test_accuracy[i] <- sum(diag(table(dat_test[, y_var], dat_test$pred))) / sum(table(dat_test[, y_var], dat_test$pred))
    grid$test_sensitivity[i] <- sensitivity(table(dat_test$pred, dat_test[, y_var]))
    grid$test_specificity[i] <- specificity(table(dat_test$pred, dat_test[, y_var]))
    
  }
  
  # Store best parameters and results arranged by test accuracy
  gs_results <<- grid %>% dplyr::arrange(desc(test_accuracy)) %>% head(1)
  row.names(gs_results) <<- as.character(y_var)
  
}

# Store RF parameters and results
start_time <- Sys.time()
best_parameters <- list()
for (var in y_cols) {
  cat(paste0("Variable: ", var, "\n")) # print current variable
  grid_search_rf(y_var = var, dat = dat, grid = hyper_grid)
  best_parameters[[length(best_parameters) + 1]] <- gs_results
}
end_time <- Sys.time()
end_time - start_time

# Store results in data frame
results <- as.data.frame(do.call(rbind, best_parameters))

rm(gs_results, best_parameters)

#===============================================================================
# Run best models and attach predicted values to original data frame

best_RF_model <- function(y_var, dat) {
  
  # Subset to non-missing variables for training model
  dat_train <- dat[which(!is.na(dat[[y_var]])), ]
  
  # Keep full data for predictions
  dat_test <- dat
  
  model <- ranger(
    formula         = dat_train[[y_var]]~var1+var2+var3+var4+var5,
    data            = dat_train, 
    classification  = TRUE,
    num.trees       = 500,
    mtry            = results[y_var, ]$mtry,
    min.node.size   = results[y_var, ]$node_size,
    sample.fraction = results[y_var, ]$samp_size,
    importance      = 'permutation',
    seed            = 123
  )
  
  # Predict on full data
  pred <- predict(model, dat_test)
  dat_test$pred <- pred$predictions
  
  cat(paste0("Variable: ", y_var, '\n')) # print current variable
  
  # Accuracy on test data
  test <- dat_test[which(!is.na(dat_test[[y_var]])), ]
  cat(paste0('Accuracy on Test Data: ',
             sum(diag(table(test[, y_var], test$pred))) / sum(table(test[, y_var], test$pred)),
             '\nConfusion Matrix:'))
  
  # Check that predictions line up reasonably well with train data set
  print(table(test[, y_var], test$pred))
  
  # Assign predictions and variable importance to global environment
  var_pred <<- pred$predictions
  var_imp <<- model$variable.importance
  
}

# Get best model predictions and variable importance for each y variable
var_pred_list <- list()
var_imp_list <- list()
for (var in y_cols) {
  best_RF_model(y_var = var, dat = dat)
  cat(paste0('--------------------\n'))
  var_pred_list[[length(var_pred_list) + 1]] <- var_pred
  var_imp_list[[length(var_imp_list) + 1]] <- var_imp
}
names(var_pred_list) <- y_cols
names(var_imp_list) <- y_cols

# Store variable importance in data frame
var_pred_dat <- as.data.frame(do.call(cbind, var_pred_list)) - 1
names(var_pred_dat) <- paste0(colnames(var_pred_dat), '_pred')
var_imp_dat <- as.data.frame(do.call(cbind, var_imp_list))

# Join predicted values to original data frame
dat <- cbind(dat, var_pred_dat)

rm(var_pred_list, var_imp_list)



