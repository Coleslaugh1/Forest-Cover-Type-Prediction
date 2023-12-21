# Setting up work environment and libraries -------------------------------

setwd(dir = "C:/Users/camer/Documents/Stat 348/Forest-Cover-Type-Prediction/")

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(discrim)
library(themis)
library(stacks)
library(kernlab)
library(keras)
library(lightgbm)
library(bonsai)
library(dbarts)

rawdata <- vroom(file = "train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))
test_input <- vroom(file = "test.csv")

my_recipe <- recipe(Cover_Type ~ ., data = rawdata) %>%
  update_role(Id, new_role="id") %>% 
  step_rm(Id) %>% 
  step_normalize(all_numeric_predictors())

prep_recipe <- prep(my_recipe)
baked_data <- bake(prep_recipe, new_data = rawdata)

format_and_write <- function(predictions, file){
  final_preds <- predictions %>%
    mutate(Cover_Type = .pred_class) %>%
    mutate(Id = test_input$Id) %>%
    dplyr::select(Id, Cover_Type)
  
  vroom_write(final_preds,file,delim = ",")
  #save(file="./MyFile.RData", list=c("object1", "object2",...))
}

# dar ---------------------------------------------------------------------

library(klaR)
dar_model <- discrim_regularized(frac_common_cov = tune(),
                                 frac_identity = tune()) %>%
  set_mode("classification") %>%
  set_engine("klaR")

dar_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(dar_model)

tuning_grid <- grid_regular(frac_common_cov(),
                            frac_identity(),
                            levels = 4)

folds <- vfold_cv(rawdata, v = 5, repeats=1)

# cl <- makePSOCKcluster(4)
# registerDoParallel(cl)
CV_results <- dar_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))
# stopCluster(cl)

bestTune <- CV_results %>%
  select_best("accuracy")

# final_dal_wf <-
#   dal_workflow %>%
#   fit(data=rawdata)

final_dar_wf <-
  dar_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=rawdata)

dar_predictions <- final_dar_wf %>%
  predict(new_data = test_input, type="class")

format_and_write(dar_predictions, "dar_preds.csv")

# bart --------------------------------------------------------------------

bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

tuning_grid <- grid_regular(trees(),
                            levels=4)

folds <- vfold_cv(rawdata, v = 15, repeats=1)

#cl <- makePSOCKcluster(10)
#registerDoParallel(cl)
CV_results <- bart_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))
#stopCluster(cl)

bestTune <- CV_results %>%
  select_best("accuracy")

final_bart_wf <-
  bart_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=rawdata)


bart_predictions <- final_bart_wf %>%
  predict(new_data = test_input, type="class")

format_and_write(bart_predictions, "bart_preds.csv")

# stack -------------------------------------------------------------------

untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

folds <- vfold_cv(rawdata, v = 5, repeats = 1)

rf_recipe <- recipe(Cover_Type~., data=rawdata) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

rf_mod <- rand_forest(min_n = 1, mtry = 15, trees = 500) %>%
  set_engine('ranger') %>%
  set_mode('classification')

rf_wf <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_recipe)

rf_model <- fit_resamples(rf_wf,
                          resamples = folds,
                          metrics = metric_set(roc_auc),
                          control = tuned_model)

nn_recipe <- recipe(Cover_Type~., data = rawdata) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

nn_model <- mlp(hidden_units = 10,
                epochs = 50) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)

nn_model <- fit_resamples(nn_wf,
                          resamples = folds,
                          metrics = metric_set(roc_auc),
                          control = tuned_model)

boost_recipe <- recipe(Cover_Type~., data=rawdata) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

boost_mod <- boost_tree(trees = 500, learn_rate = .01, tree_depth = 2) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

boost_wf <- workflow() %>%
  add_model(boost_mod) %>%
  add_recipe(boost_recipe)


boost_model <- fit_resamples(boost_wf,
                             resamples = folds,
                             metrics = metric_set(roc_auc),
                             control = tuned_model)

my_stack <- stacks() %>%
  add_candidates(rf_model) %>%
  add_candidates(nn_model) %>%
  add_candidates(boost_model)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

stack_preds <- stack_mod %>%
  predict(new_data = test_input, type = "class")

format_and_write(stack_preds, "stack_preds.csv")
