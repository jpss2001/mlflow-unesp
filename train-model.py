# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select * from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

# DBTITLE 1,Imports
#import das libs
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

import mlflow

#import dos dados
sdf = spark.table ("sandbox_apoiadores.abt_dota_pre_match")
df = sdf.toPandas()

# COMMAND ----------

# DBTITLE 1,Definição das variáveis
target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list(set(df.columns.tolist()) - set([target_column,id_column]))

Y = df[target_column]
X = df[features_columns]


# COMMAND ----------

# DBTITLE 1,Split test e train
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=0.2, random_state=42)

print("numero de linhas em X_train:", X_train.shape[0])
print("numero de linhas em X_test:", X_test.shape[0])
print("numero de linhas em Y_train:", Y_train.shape[0])
print("numero de linhas em Y_test:", Y_test.shape[0])

# COMMAND ----------

# DBTITLE 1,Setup do experimento mlflow
mlflow.set_experiment("/Users/joao.ps.silva@unesp.br/dota_unesp_joaoP")

# COMMAND ----------

# DBTITLE 1,Run do experimento
with mlflow.start_run():
    
    mlflow.sklearn.autolog()
    
    # model = tree.DecisionTreeClassifier()
    # model.fit(X_train,Y_train)
    
    model = ensemble.RandomForestClassifier(n_estimators=100)
    model.fit(X_train,Y_train)

    # model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.7)
    # model.fit(X_train,Y_train)

    Y_train_pred = model.predict(X_train)
    Y_train_prob = model.predict_proba(X_train)

    acc_train = metrics.accuracy_score(Y_train, Y_train_pred)

    print("Acuracia em treino:", acc_train)

    Y_test_pred = model.predict(X_test)
    Y_test_prob = model.predict_proba(X_test)

    acc_test = metrics.accuracy_score(Y_test, Y_test_pred)

    print("Acuracia em treino:", acc_test)
    
   # mlflow.log_dict({"features": X_train.columns.tolist()})
