from carla.plotting.plotting import summary_plot, single_sample_plot
from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods import GrowingSpheres
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog import CsvCatalog, mnist_bin
from carla.models.catalog.train_model import train_model
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def cchvae(ml_model,dataset ):
    hyperparams = {
            "data_name": 'mnist_bin',
            "n_search_samples": 100,
            "p_norm": 1,
            "step": 0.1,
            "max_iter": 1000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [784, 512, 256, 8],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 32,
            },
        }

    # define your recourse method
    recourse_method = recourse_catalog.CCHVAE(ml_model, hyperparams)

    # get some negative instances
    factuals = predict_negative_instances(ml_model, dataset.df_test)
    factuals = factuals[:100]
    factuals.drop(factuals.columns[[0,-1]],axis=1,inplace=True)

    # find counterfactuals
    counterfactuals = recourse_method.get_counterfactuals(factuals)

    factuals = factuals.to_numpy()
    counterfactuals = counterfactuals.to_numpy()

    return factuals, counterfactuals

def wacher(ml_model, dataset):
    hyperparams = {
        "feature_cost": np.ones((dataset.df_train.shape[1]-2)),
        "lr": 0.001,
        "lambda_": 0.01,
        "n_iter": 1000,
        "t_max_min": 0.5,
        "norm": 1,
        "clamp": True,
        "loss_type": "BCE",
        "y_target": [0, 1],
        "binary_cat_features": False,
    }

    # define your recourse method
    recourse_method = recourse_catalog.Wachter(ml_model, hyperparams)

    # get some negative instances
    factuals = predict_negative_instances(ml_model, dataset.df_test)
    factuals = factuals[:100]
    factuals.drop(factuals.columns[[0,-1]],axis=1,inplace=True)

    # find counterfactuals
    counterfactuals = recourse_method.get_counterfactuals(factuals)

    factuals = factuals.to_numpy()
    counterfactuals = counterfactuals.to_numpy()

    return factuals, counterfactuals

def revise(ml_model, dataset):
    hyperparams = {
        "data_name": 'mnist',
        "lambda": 0.5,
        "optimizer": "adam",
        "lr": 0.1,
        "max_iter": 1000,
        "target_class": [0, 1],
        "binary_cat_features": True,
        "vae_params": {
            "layers": [784,400,20],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 128,
        },
    }

    # define your recourse method
    recourse_method = recourse_catalog.Revise(ml_model,dataset, hyperparams)

    # get some negative instances
    factuals = predict_negative_instances(ml_model, dataset.df_test)
    factuals = factuals[:100]
    factuals.drop(factuals.columns[[0,-1]],axis=1,inplace=True)

    # find counterfactuals
    counterfactuals = recourse_method.get_counterfactuals(factuals)

    factuals = factuals.to_numpy()
    counterfactuals = counterfactuals.to_numpy()

    return factuals, counterfactuals

def clue(ml_model, dataset):
    hyperparams = {
        "data_name": 'mnist',
        "train_vae": True,
        "width": 200,
        "depth": 2,
        "latent_dim": 20,
        "batch_size": 128,
        "epochs": 5,
        "lr": 0.001,
        "early_stop": 10,
    }

    # define your recourse method
    recourse_method = recourse_catalog.Clue(dataset,ml_model, hyperparams)

    # get some negative instances
    factuals = predict_negative_instances(ml_model, dataset.df_test)
    factuals = factuals[:10]
    factuals.drop(factuals.columns[[0,-1]],axis=1,inplace=True)

    # find counterfactuals
    counterfactuals = recourse_method.get_counterfactuals(factuals)

    factuals = factuals.to_numpy()
    counterfactuals = counterfactuals.to_numpy()

    return factuals, counterfactuals

def crud(ml_model, dataset):
    hyperparams = {
        "data_name": 'mnist',
        "target_class": [0, 1],
        "lambda_param": 0.001,
        "optimizer": "RMSprop",
        "lr": 0.008,
        "max_iter": 2000,
        "binary_cat_features": True,
        "vae_params": {
            "layers": [784, 400, 20],
            "train": True,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

    # define your recourse method
    recourse_method = recourse_catalog.CRUD(ml_model, hyperparams)

    # get some negative instances
    factuals = predict_negative_instances(ml_model, dataset.df_test)
    factuals = factuals[:10]
    factuals.drop(factuals.columns[[0]],axis=1,inplace=True)

    # find counterfactuals
    counterfactuals = recourse_method.get_counterfactuals(factuals)

    factuals = factuals.to_numpy()
    counterfactuals = counterfactuals.to_numpy()

    return factuals[:,:784], counterfactuals

def dice(ml_model, dataset):
    hyperparams = {"num": 1, "desired_class": 1, "posthoc_sparsity_param": 0.1}

    # define your recourse method
    recourse_method = recourse_catalog.Dice(ml_model, hyperparams)

    # get some negative instances
    factuals = predict_negative_instances(ml_model, dataset.df_test)
    factuals = factuals[:10]
    factuals.drop(factuals.columns[[0,-1]],axis=1,inplace=True)

    # find counterfactuals
    counterfactuals = recourse_method.get_counterfactuals(factuals)

    factuals = factuals.to_numpy()
    counterfactuals = counterfactuals.to_numpy()

    return factuals, counterfactuals


def face(ml_model, dataset):
    hyperparams = {"mode": 'knn', "fraction": 0.1}

    # define your recourse method
    recourse_method = recourse_catalog.Face(ml_model, hyperparams)

    # get some negative instances
    factuals = predict_negative_instances(ml_model, dataset.df_test)
    factuals = factuals[:10]
    factuals.drop(factuals.columns[[0,-1]],axis=1,inplace=True)

    # find counterfactuals
    counterfactuals = recourse_method.get_counterfactuals(factuals)

    factuals = factuals.to_numpy()
    counterfactuals = counterfactuals.to_numpy()

    return factuals, counterfactuals



########################################################################################################################

dataset = mnist_bin()


# data_name = "adult"
# dataset = OnlineCatalog(data_name)

# load catalog model
model_type = "ann"
ml_model = MLModelCatalog(
    dataset,
    model_type=model_type,
    load_online=False,
    backend="pytorch"
)

# ml_model._model = train_model(catalog_model=ml_model,
#                        x_train=dataset.df_train[dataset.continuous],
#                        y_train=dataset.df_train[dataset.target],
#                        x_test=dataset.df_test[dataset.continuous],
#                        y_test=dataset.df_test[dataset.target],
#                        learning_rate=0.0001,
#                        epochs=50,
#                        batch_size=128,
#                        hidden_size=[400],
#                        n_estimators=0,
#                        max_depth=0)

ml_model.train(learning_rate=0.0001,epochs=50,batch_size=128,force_train=False,hidden_size=[400])

##################
# factuals, counterfactuals = cchvae(ml_model, dataset)
factuals, counterfactuals = wacher(ml_model, dataset)
# factuals, counterfactuals = revise(ml_model, dataset)
# factuals, counterfactuals = clue(ml_model, dataset)
# factuals, counterfactuals = crud(ml_model, dataset)
# factuals, counterfactuals = dice(ml_model, dataset)
# factuals, counterfactuals = face(ml_model, dataset)


######################


num_sample_toshow = 10
fig, axs = plt.subplots(2, num_sample_toshow, figsize=(21, 11))
for i in range(num_sample_toshow):
    axs[0][i].imshow(np.reshape(factuals[i,:],(28, 28)), cmap='gray')
    axs[1][i].imshow(np.reshape(counterfactuals[i,:],(28, 28)), cmap='gray')

fig.show()
plt.draw()
plt.tight_layout()
plt.waitforbuttonpress()


import pdb
pdb.set_trace()


########################################################################################################################









# remove rows where no counterfactual was found
counterfactuals = counterfactuals.dropna()
factuals = factuals[factuals.index.isin(counterfactuals.index)]

single_sample_plot(factuals.iloc[0], counterfactuals.iloc[0], dataset)

summary_plot(factuals, counterfactuals, dataset, topn=3)

import pdb
pdb.set_trace()

print('---------------')