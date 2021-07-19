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

grid_search_rf <- function(y_var, dat){
  
  # Subset to non-missing y variables
  dat <- dat[which(!is.na(dat[[y_var]])), ]
  
  # Hyperparameter grid search
  hyper_grid <- expand.grid(
    mtry       = seq(2, 9, by = 1),
    node_size  = seq(1, 13, by = 4),
    sampe_size = c(.625, .75, .875)
  )
  
  for(i in 1:nrow(hyper_grid)) {
    
    # print current iteration
    cat(paste0('Iteration: ', i, ' of ', nrow(hyper_grid)), '\n')
    
    # train model
    model <- ranger(
      formula         = dat[[y_var]]~var1+var2+var3+var4+var5,
      data            = dat,
      classification  = TRUE,
      num.trees       = 500,
      mtry            = hyper_grid$mtry[i],
      min.node.size   = hyper_grid$node_size[i],
      sample.fraction = hyper_grid$sampe_size[i],
      seed            = 123
    )
    
    # add percent correct to grid
    hyper_grid$rf_pct_correct[i] <- (1 - model$prediction.error)
    
    # add sensitivity/specificity
    hyper_grid$rf_sensitivity[i] <- (model$confusion.matrix[1,1] / sum(model$confusion.matrix[1,]))
    hyper_grid$rf_specificity[i] <- (model$confusion.matrix[2,2] / sum(model$confusion.matrix[2,]))
  }
  
  hyper_grid %>% 
    dplyr::arrange(desc(rf_pct_correct)) %>%
    head(10)
  
  # Store best parameters and results
  gs_results <<- hyper_grid %>% dplyr::arrange(desc(rf_pct_correct)) %>% head(1)
  row.names(gs_results) <<- as.character(y_var)
  
}

# Store RF parameters and results
start_time <- Sys.time()
best_parameters <- list()
for (var in y_cols) {
  cat(paste0("Variable: ", var, "\n")) # print current variable
  grid_search_rf(y_var = var, dat = dat)
  best_parameters[[length(best_parameters) + 1]] <- gs_results
}
end_time <- Sys.time()
end_time - start_time

# Store parameters and results of best models in data frame
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
    sample.fraction = results[y_var, ]$sampe_size,
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
var_pred_dat <- as.data.frame(do.call(cbind, var_pred_list))
names(var_pred_dat) <- paste0(colnames(var_pred_dat), '_pred')
var_imp_dat <- as.data.frame(do.call(cbind, var_imp_list))

# Join predicted values to original data frame
dat <- cbind(dat, var_pred_dat)

rm(var_pred_list, var_imp_list)



