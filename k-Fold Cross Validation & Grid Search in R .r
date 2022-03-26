{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b097a1d2",
   "metadata": {
    "papermill": {
     "duration": 0.045021,
     "end_time": "2022-03-26T15:05:16.201063",
     "exception": false,
     "start_time": "2022-03-26T15:05:16.156042",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **k-Fold Cross Validation & Grid Search in R**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "45c95f9d",
   "metadata": {
    "papermill": {
     "duration": 0.041797,
     "end_time": "2022-03-26T15:05:16.288542",
     "exception": false,
     "start_time": "2022-03-26T15:05:16.246745",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Importing the dataset**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3ce00a1f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-03-26T15:05:16.379144Z",
     "iopub.status.busy": "2022-03-26T15:05:16.376214Z",
     "iopub.status.idle": "2022-03-26T15:05:16.538960Z",
     "shell.execute_reply": "2022-03-26T15:05:16.539513Z"
    },
    "papermill": {
     "duration": 0.208364,
     "end_time": "2022-03-26T15:05:16.539759",
     "exception": false,
     "start_time": "2022-03-26T15:05:16.331395",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 6 × 3</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>Age</th><th scope=col>EstimatedSalary</th><th scope=col>Purchased</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>1</th><td>19</td><td>19000</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>2</th><td>35</td><td>20000</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>3</th><td>26</td><td>43000</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>4</th><td>27</td><td>57000</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>5</th><td>19</td><td>76000</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>6</th><td>27</td><td>58000</td><td>0</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 6 × 3\n",
       "\\begin{tabular}{r|lll}\n",
       "  & Age & EstimatedSalary & Purchased\\\\\n",
       "  & <int> & <int> & <int>\\\\\n",
       "\\hline\n",
       "\t1 & 19 & 19000 & 0\\\\\n",
       "\t2 & 35 & 20000 & 0\\\\\n",
       "\t3 & 26 & 43000 & 0\\\\\n",
       "\t4 & 27 & 57000 & 0\\\\\n",
       "\t5 & 19 & 76000 & 0\\\\\n",
       "\t6 & 27 & 58000 & 0\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 6 × 3\n",
       "\n",
       "| <!--/--> | Age &lt;int&gt; | EstimatedSalary &lt;int&gt; | Purchased &lt;int&gt; |\n",
       "|---|---|---|---|\n",
       "| 1 | 19 | 19000 | 0 |\n",
       "| 2 | 35 | 20000 | 0 |\n",
       "| 3 | 26 | 43000 | 0 |\n",
       "| 4 | 27 | 57000 | 0 |\n",
       "| 5 | 19 | 76000 | 0 |\n",
       "| 6 | 27 | 58000 | 0 |\n",
       "\n"
      ],
      "text/plain": [
       "  Age EstimatedSalary Purchased\n",
       "1 19  19000           0        \n",
       "2 35  20000           0        \n",
       "3 26  43000           0        \n",
       "4 27  57000           0        \n",
       "5 19  76000           0        \n",
       "6 27  58000           0        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "ds = read.csv('../input/social-network-ads/Social_Network_Ads.csv')\n",
    "ds = ds[3:5]\n",
    "head(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a81cea4",
   "metadata": {
    "papermill": {
     "duration": 0.043256,
     "end_time": "2022-03-26T15:05:16.627741",
     "exception": false,
     "start_time": "2022-03-26T15:05:16.584485",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Encoding the target feature as factor**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4b46644b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-03-26T15:05:16.756166Z",
     "iopub.status.busy": "2022-03-26T15:05:16.717932Z",
     "iopub.status.idle": "2022-03-26T15:05:16.769322Z",
     "shell.execute_reply": "2022-03-26T15:05:16.767622Z"
    },
    "papermill": {
     "duration": 0.098843,
     "end_time": "2022-03-26T15:05:16.769478",
     "exception": false,
     "start_time": "2022-03-26T15:05:16.670635",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "ds$Purchased = factor(ds$Purchased, levels = c(0, 1))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7debf44b",
   "metadata": {
    "papermill": {
     "duration": 0.04245,
     "end_time": "2022-03-26T15:05:16.854596",
     "exception": false,
     "start_time": "2022-03-26T15:05:16.812146",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Splitting the data set & Feature scaling**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e00603ed",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-03-26T15:05:16.946722Z",
     "iopub.status.busy": "2022-03-26T15:05:16.944644Z",
     "iopub.status.idle": "2022-03-26T15:05:17.018602Z",
     "shell.execute_reply": "2022-03-26T15:05:17.016925Z"
    },
    "papermill": {
     "duration": 0.121169,
     "end_time": "2022-03-26T15:05:17.018750",
     "exception": false,
     "start_time": "2022-03-26T15:05:16.897581",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Splitting the dataset into the Training set and Test set\n",
    "# install.packages('caTools')\n",
    "library(caTools)\n",
    "set.seed(123)\n",
    "split = sample.split(ds$Purchased, SplitRatio = 0.75)\n",
    "train_set = subset(ds, split == TRUE)\n",
    "test_set = subset(ds, split == FALSE)\n",
    "\n",
    "# Feature Scaling\n",
    "train_set[-3] = scale(train_set[-3])\n",
    "test_set[-3] = scale(test_set[-3])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8cc0acbb",
   "metadata": {
    "papermill": {
     "duration": 0.043311,
     "end_time": "2022-03-26T15:05:17.104932",
     "exception": false,
     "start_time": "2022-03-26T15:05:17.061621",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Fitting Kernel SVM to the Train set & Predicting the Test set**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "abf678fc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-03-26T15:05:17.200894Z",
     "iopub.status.busy": "2022-03-26T15:05:17.198865Z",
     "iopub.status.idle": "2022-03-26T15:05:17.300906Z",
     "shell.execute_reply": "2022-03-26T15:05:17.299537Z"
    },
    "papermill": {
     "duration": 0.153086,
     "end_time": "2022-03-26T15:05:17.301072",
     "exception": false,
     "start_time": "2022-03-26T15:05:17.147986",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " The confusion matrix for Kernel SVM model is: \n",
      " \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "   y_pred\n",
       "     0  1\n",
       "  0 58  6\n",
       "  1  4 32"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Fitting Kernel SVM to the Training set\n",
    "# install.packages('e1071')\n",
    "library(e1071)\n",
    "classifier = svm(formula = Purchased ~ .,\n",
    "                 data = train_set,\n",
    "                 type = 'C-classification',\n",
    "                 kernel = 'radial')\n",
    "\n",
    "# Predicting the Test set results\n",
    "y_pred = predict(classifier, newdata = test_set[-3])\n",
    "\n",
    "# Making the Confusion Matrix\n",
    "cm = table(test_set[, 3], y_pred)\n",
    "cat('\\n The confusion matrix for Kernel SVM model is: \\n \\n')\n",
    "cm"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09299c83",
   "metadata": {
    "papermill": {
     "duration": 0.045167,
     "end_time": "2022-03-26T15:05:17.392381",
     "exception": false,
     "start_time": "2022-03-26T15:05:17.347214",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Evaluation Metrics**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "21840228",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-03-26T15:05:17.489562Z",
     "iopub.status.busy": "2022-03-26T15:05:17.487600Z",
     "iopub.status.idle": "2022-03-26T15:05:17.542159Z",
     "shell.execute_reply": "2022-03-26T15:05:17.540498Z"
    },
    "papermill": {
     "duration": 0.104529,
     "end_time": "2022-03-26T15:05:17.542304",
     "exception": false,
     "start_time": "2022-03-26T15:05:17.437775",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Accuracy of Kernel SVM  Model is: 0.9\n",
      " \n",
      "The Evaluation Metrics of Kernel SVM  Model is: \n",
      " \n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 2 × 3</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>precision</th><th scope=col>recall</th><th scope=col>f1</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>0</th><td>0.9354839</td><td>0.9062500</td><td>0.9206349</td></tr>\n",
       "\t<tr><th scope=row>1</th><td>0.8421053</td><td>0.8888889</td><td>0.8648649</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 2 × 3\n",
       "\\begin{tabular}{r|lll}\n",
       "  & precision & recall & f1\\\\\n",
       "  & <dbl> & <dbl> & <dbl>\\\\\n",
       "\\hline\n",
       "\t0 & 0.9354839 & 0.9062500 & 0.9206349\\\\\n",
       "\t1 & 0.8421053 & 0.8888889 & 0.8648649\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 2 × 3\n",
       "\n",
       "| <!--/--> | precision &lt;dbl&gt; | recall &lt;dbl&gt; | f1 &lt;dbl&gt; |\n",
       "|---|---|---|---|\n",
       "| 0 | 0.9354839 | 0.9062500 | 0.9206349 |\n",
       "| 1 | 0.8421053 | 0.8888889 | 0.8648649 |\n",
       "\n"
      ],
      "text/plain": [
       "  precision recall    f1       \n",
       "0 0.9354839 0.9062500 0.9206349\n",
       "1 0.8421053 0.8888889 0.8648649"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "n = sum(cm) # number of instances\n",
    "nc = nrow(cm) # number of classes\n",
    "diag = diag(cm) # number of correctly classified instances per class \n",
    "rowsums = apply(cm, 1, sum) # number of instances per class\n",
    "colsums = apply(cm, 2, sum) # number of predictions per class\n",
    "p = rowsums / n # distribution of instances over the actual classes\n",
    "q = colsums / n # distribution of instances over the predicted classes\n",
    "accuracy = sum(diag) / n \n",
    "cat(\"\\n Accuracy of Kernel SVM  Model is:\", accuracy)  \n",
    "precision = diag / colsums \n",
    "recall = diag / rowsums \n",
    "f1 = 2 * precision * recall / (precision + recall)\n",
    "cat(\"\\n \\nThe Evaluation Metrics of Kernel SVM  Model is: \\n \\n\")\n",
    "data.frame(precision, recall, f1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47ffb2ae",
   "metadata": {
    "papermill": {
     "duration": 0.051672,
     "end_time": "2022-03-26T15:05:17.645362",
     "exception": false,
     "start_time": "2022-03-26T15:05:17.593690",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Applying k-Fold Cross Validation**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "616752ef",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-03-26T15:05:17.751138Z",
     "iopub.status.busy": "2022-03-26T15:05:17.749459Z",
     "iopub.status.idle": "2022-03-26T15:05:20.416891Z",
     "shell.execute_reply": "2022-03-26T15:05:20.414786Z"
    },
    "papermill": {
     "duration": 2.721924,
     "end_time": "2022-03-26T15:05:20.417060",
     "exception": false,
     "start_time": "2022-03-26T15:05:17.695136",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Loading required package: lattice\n",
      "\n",
      "Loading required package: ggplot2\n",
      "\n",
      "\n",
      "Attaching package: ‘caret’\n",
      "\n",
      "\n",
      "The following object is masked from ‘package:httr’:\n",
      "\n",
      "    progress\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Accuracy of Kernel SVM k-Fold Cross Validated  Model is: 0.9162848"
     ]
    }
   ],
   "source": [
    "# install.packages('caret')\n",
    "library(caret)\n",
    "folds = createFolds(train_set$Purchased, k = 10)\n",
    "cv = lapply(folds, function(x) {\n",
    "  training_fold = train_set[-x, ]\n",
    "  test_fold = train_set[x, ]\n",
    "  classifier = svm(formula = Purchased ~ .,\n",
    "                   data = training_fold,\n",
    "                   type = 'C-classification',\n",
    "                   kernel = 'radial')\n",
    "  y_pred = predict(classifier, newdata = test_fold[-3])\n",
    "  cm = table(test_fold[, 3], y_pred)\n",
    "  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])\n",
    "  return(accuracy)\n",
    "})\n",
    "accuracy = mean(as.numeric(cv))\n",
    "cat(\"\\nAccuracy of Kernel SVM k-Fold Cross Validated  Model is:\", accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e4f6abb",
   "metadata": {
    "papermill": {
     "duration": 0.054997,
     "end_time": "2022-03-26T15:05:20.528422",
     "exception": false,
     "start_time": "2022-03-26T15:05:20.473425",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Applying Grid Search to find the best parameters**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a757cfc5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-03-26T15:05:20.647993Z",
     "iopub.status.busy": "2022-03-26T15:05:20.645323Z",
     "iopub.status.idle": "2022-03-26T15:05:24.879512Z",
     "shell.execute_reply": "2022-03-26T15:05:24.876914Z"
    },
    "papermill": {
     "duration": 4.295735,
     "end_time": "2022-03-26T15:05:24.879740",
     "exception": false,
     "start_time": "2022-03-26T15:05:20.584005",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Support Vector Machines with Radial Basis Function Kernel \n",
       "\n",
       "300 samples\n",
       "  2 predictor\n",
       "  2 classes: '0', '1' \n",
       "\n",
       "No pre-processing\n",
       "Resampling: Bootstrapped (25 reps) \n",
       "Summary of sample sizes: 300, 300, 300, 300, 300, 300, ... \n",
       "Resampling results across tuning parameters:\n",
       "\n",
       "  C     Accuracy   Kappa    \n",
       "  0.25  0.9145693  0.8130036\n",
       "  0.50  0.9159184  0.8157252\n",
       "  1.00  0.9186723  0.8215380\n",
       "\n",
       "Tuning parameter 'sigma' was held constant at a value of 1.327355\n",
       "Accuracy was used to select the optimal model using the largest value.\n",
       "The final values used for the model were sigma = 1.327355 and C = 1."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 1 × 2</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>sigma</th><th scope=col>C</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>3</th><td>1.327355</td><td>1</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 1 × 2\n",
       "\\begin{tabular}{r|ll}\n",
       "  & sigma & C\\\\\n",
       "  & <dbl> & <dbl>\\\\\n",
       "\\hline\n",
       "\t3 & 1.327355 & 1\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 1 × 2\n",
       "\n",
       "| <!--/--> | sigma &lt;dbl&gt; | C &lt;dbl&gt; |\n",
       "|---|---|---|\n",
       "| 3 | 1.327355 | 1 |\n",
       "\n"
      ],
      "text/plain": [
       "  sigma    C\n",
       "3 1.327355 1"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# install.packages('caret')\n",
    "library(caret)\n",
    "classifier = train(form = Purchased ~ ., data = train_set, method = 'svmRadial')\n",
    "classifier\n",
    "classifier$bestTune"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a395f44",
   "metadata": {
    "papermill": {
     "duration": 0.061973,
     "end_time": "2022-03-26T15:05:25.038338",
     "exception": false,
     "start_time": "2022-03-26T15:05:24.976365",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30353453",
   "metadata": {
    "papermill": {
     "duration": 0.059969,
     "end_time": "2022-03-26T15:05:25.158830",
     "exception": false,
     "start_time": "2022-03-26T15:05:25.098861",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "R",
   "language": "R",
   "name": "ir"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "4.0.5"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 12.509447,
   "end_time": "2022-03-26T15:05:25.329713",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-03-26T15:05:12.820266",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
