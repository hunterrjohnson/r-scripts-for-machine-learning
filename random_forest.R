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

# Store results in data frame
results <- as.data.frame(do.call(rbind, best_parameters))



